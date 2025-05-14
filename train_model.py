import unsloth
import os
import torch
import logging
import time
import transformers
import trl
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import glob
import numpy as np
import re
import logging
import json

#torch.set_num_threads(6)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 打印库版本
logger.info(f"transformers version: {transformers.__version__}")
logger.info(f"trl version: {trl.__version__}")
logger.info(f"unsloth version: {unsloth.__version__}")

# 常量
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_NAME = "Jofthomas/hermes-function-calling-thinking-V1"
OUTPUT_DIR = os.path.join(".", "fine_tuned_model")
MAX_SEQ_LENGTH = 2048
LORA_RANK = 16
LORA_ALPHA = 32
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
EPOCHS = 3
VALIDATION_SPLIT = 0.005
EVAL_SAMPLES = 10

def load_model_and_tokenizer(max_retries=3, retry_delay=5):
    logger.info(f"加载模型 {MODEL_NAME}")
    for attempt in range(1, max_retries + 1):
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                resume_download=True
            )
            logger.info("模型和分词器加载成功")

            #model.config.model_type = "qwen2"
            #model.config.vocab_size = len(tokenizer)

            tokenizer = get_chat_template(
                tokenizer,
                chat_template="qwen-2.5",
            )

            logger.info("应用 LoRA 适配器")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                modules_to_save=["lm_head", "embed_tokens"],
                use_gradient_checkpointing=True,
                random_state=3407,
                max_seq_length=MAX_SEQ_LENGTH
            )
            logger.info("LoRA 适配器应用成功")

            # Explicitly set save_embedding_layers=True in the PeftConfig
            if hasattr(model, "peft_config"):
                for adapter_name, peft_config in model.peft_config.items():
                    peft_config.save_embedding_layers = True
                    logger.info(f"Set save_embedding_layers=True for adapter {adapter_name}")

            return model, tokenizer
        except Exception as e:
            logger.error(f"尝试 {attempt} 失败: {str(e)}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)

def prepare_dataset():
    logger.info(f"加载数据集 {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"数据集样本: {dataset[0]}")
    dataset = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=3407)
    return dataset["train"], dataset["test"]

logger = logging.getLogger(__name__)

def preprocess_dataset(dataset: Dataset, tokenizer, max_seq_length: int = 2048):
    def format_conversation(examples):
        input_texts = []
        label_texts = []
        skipped_count = 0

        for idx, example in enumerate(examples["conversations"]):
            if not isinstance(example, list) or not example:
                logger.warning(f"样本 {idx} 无效的对话格式: {example}")
                skipped_count += 1
                continue

            # 检查样本内容
            logger.debug(f"样本 {idx} 原始内容: {example}")

            # 转换角色：human -> user, model -> assistant
            role_mapping = {
                "human": "user",
                "model": "assistant",
                "assistant": "assistant",
                "user": "user",
                "system": "system",
                "tool": "tool"
            }
            try:
                example = [
                    {"role": role_mapping.get(msg["role"], msg["role"]), "content": msg["content"]}
                    for msg in example
                ]
            except KeyError as e:
                logger.error(f"样本 {idx} 缺少 role 或 content: {example}")
                skipped_count += 1
                continue

            # 检查转换后的角色
            roles = set(msg["role"] for msg in example)
            logger.debug(f"样本 {idx} 转换后角色: {roles}")
            if any(role not in ["system", "user", "assistant", "tool"] for role in roles):
                logger.warning(f"样本 {idx} 包含不受支持的角色: {roles}")
                skipped_count += 1
                continue

            found_assistant = False
            for i in range(len(example)):
                if example[i]["role"] == "assistant":
                    found_assistant = True
                    input_messages = example[:i]
                    label_message = example[i]

                    try:
                        # 检查消息内容
                        logger.debug(f"样本 {idx} 输入消息: {input_messages}")
                        logger.debug(f"样本 {idx} 标签消息: {label_message}")

                        input_text = tokenizer.apply_chat_template(
                            input_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        label_text = label_message['content'] + "\n<|im_end|>"

                        if not input_text or not label_text:
                            logger.warning(f"样本 {idx} 生成空文本: input_text={input_text}, label_text={label_text}")
                            skipped_count += 1
                            continue

                        input_texts.append(input_text)
                        label_texts.append(label_text)
                        logger.debug(f"样本 {idx} 生成成功: input_text={input_text[:50]}..., label_text={label_text[:50]}...")
                    except Exception as e:
                        logger.error(f"样本 {idx} 应用聊天模板出错: {str(e)}")
                        logger.error(f"输入消息: {input_messages}")
                        logger.error(f"标签消息: {label_message}")
                        skipped_count += 1
                        continue

            if not found_assistant:
                logger.warning(f"样本 {idx} 无 assistant 角色: {example}")
                skipped_count += 1

        logger.info(f"跳过了 {skipped_count} 个无效样本")
        tokenized_inputs = tokenizer(
            input_texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
        tokenized_labels = tokenizer(
            label_texts,
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": tokenized_labels["input_ids"],
        }

    logger.info("预处理数据集")
    print("首条样本:", dataset[0]["conversations"]) 

    processed_dataset = dataset.map(
        format_conversation,
        batched=True,
        remove_columns=["conversations"],
        load_from_cache_file=False
    )
    processed_dataset = processed_dataset.flatten()
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["input_ids"]) > 0 and len(x["labels"]) > 0
    )

    logger.info(f"处理后的数据集大小: {len(processed_dataset)}")
    return processed_dataset

class CustomDataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.tensor(example["attention_mask"]) for example in examples]
        labels = [torch.tensor(example["labels"]) for example in examples]

        # Ensure all tensors are truncated to max_length
        input_ids = [ids[:self.max_length] for ids in input_ids]
        attention_mask = [mask[:self.max_length] for mask in attention_mask]
        labels = [lbl[:self.max_length] for lbl in labels]

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, 
            batch_first=True, 
            padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, 
            batch_first=True, 
            padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def custom_data_collator(tokenizer):
    return CustomDataCollator(tokenizer, MAX_SEQ_LENGTH)

def create_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len)
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        batch_size, seq_len, vocab_size = predictions.shape
        cross_entropy = 0.0
        total_correct = 0
        total_valid_tokens = 0

        # 逐批处理 logits
        for i in range(batch_size):
            pred = predictions[i]  # (seq_len, vocab_size)
            lbl = labels[i]       # (seq_len)

            # 创建掩码，忽略 -100 和 pad_token_id
            valid_mask = (lbl != -100) & (lbl != tokenizer.pad_token_id)
            valid_pred = pred[valid_mask]
            valid_lbl = lbl[valid_mask]

            if valid_lbl.numel() > 0:
                # 计算交叉熵
                ce = torch.nn.functional.cross_entropy(valid_pred, valid_lbl, reduction="mean")
                cross_entropy += ce.item()

                # 计算 token 准确率
                pred_tokens = torch.argmax(valid_pred, dim=-1)
                correct_tokens = (pred_tokens == valid_lbl).sum().item()
                total_correct += correct_tokens
                total_valid_tokens += valid_mask.sum().item()

        # 平均交叉熵
        cross_entropy = cross_entropy / batch_size if batch_size > 0 else 0.0
        token_accuracy = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0

        return {
            "cross_entropy": cross_entropy,
            "token_accuracy": token_accuracy,
        }
    return compute_metrics

def train(model, tokenizer, train_dataset, eval_dataset):
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        save_total_limit=50,
        fp16=False,
        bf16=True,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        packing=False,
        max_seq_length=MAX_SEQ_LENGTH,
        neftune_noise_alpha=5,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        do_eval=True,
        eval_steps=10,
        eval_strategy="steps",
        report_to="tensorboard",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        gradient_accumulation_steps=4,
    )

    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    resume_from_checkpoint = None
    if checkpoint_dirs:
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("checkpoint-")[-1]))
        logger.info(f"找到检查点: {latest_checkpoint}")
        resume_from_checkpoint = latest_checkpoint
    else:
        logger.info("未找到检查点，从头开始")

    compute_metrics = create_compute_metrics(tokenizer)
    eval_dataset = eval_dataset.select(range(min(len(eval_dataset), 10)))
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=custom_data_collator(tokenizer),
        compute_metrics=compute_metrics,  # 添加 compute_metrics
    )

    logger.info(f"Running initial evaluation with {EVAL_SAMPLES} samples")
    eval_results = trainer.evaluate()
    logger.info(f"Initial evaluation results: {eval_results}")

    logger.info("开始训练")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        logger.error(f"训练失败: {str(e)}")
        raise
    logger.info("训练完成")

    logger.info("保存最终模型")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def main():
    logger.info("开始训练流程")
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, eval_dataset = prepare_dataset()
    train_dataset = preprocess_dataset(train_dataset, tokenizer)
    eval_dataset = preprocess_dataset(eval_dataset, tokenizer)
    logger.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")
    train(model, tokenizer, train_dataset, eval_dataset)
    logger.info("训练流程成功完成")

if __name__ == "__main__":
    main()