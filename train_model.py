import unsloth
import os
import torch
import logging
import time
import transformers
import trl
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import glob
import numpy as np

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
MAX_SEQ_LENGTH = 1024
LORA_RANK = 16
LORA_ALPHA = 32
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 3
VALIDATION_SPLIT = 0.1

# Qwen 聊天模板
QWEN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'human' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'model' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'tool' %}"
    "<|im_start|>tool\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
)

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

            model.config.model_type = "qwen2"
            model.config.vocab_size = len(tokenizer)

            special_tokens = {
                "additional_special_tokens": [
                    "<tools>",
                    "</tools>",
                    "<tool_call>",
                    "</tool_call>",
                    "<tool_response>",
                    "</tool_response>"
                ]
            }
            logger.info("向分词器添加特殊 token")
            num_added_tokens = tokenizer.add_special_tokens(special_tokens)
            logger.info(f"添加了 {num_added_tokens} 个特殊 token")

            logger.info("调整模型 token 嵌入")
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"模型 token 嵌入调整为 {len(tokenizer)}")

            logger.info("设置 Qwen 聊天模板")
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE

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

def preprocess_dataset(dataset, tokenizer):
    def format_conversation(examples):
        input_texts = []
        label_texts = []
        skipped_count = 0

        for example in examples["conversations"]:
            if not isinstance(example, list) or not example:
                logger.warning(f"无效的对话格式: {example}")
                skipped_count += 1
                continue

            for i in range(len(example)):
                if example[i]["role"] == "model":
                    input_messages = example[:i]
                    label_message = [example[i]]

                    try:
                        input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)
                        label_text = tokenizer.apply_chat_template(label_message, tokenize=False)
                        input_texts.append(input_text)
                        label_texts.append(label_text)
                    except Exception as e:
                        logger.error(f"应用聊天模板出错: {str(e)}")
                        skipped_count += 1
                        continue

        logger.info(f"跳过了 {skipped_count} 个无效样本")
        return {"input_text": input_texts, "label_text": label_texts}

    logger.info("预处理数据集")
    processed_dataset = dataset.map(
        format_conversation,
        batched=True,
        remove_columns=["conversations"]
    )
    processed_dataset = processed_dataset.flatten()
    processed_dataset = processed_dataset.filter(
        lambda x: x["input_text"] is not None and x["label_text"] is not None
    )
    logger.info(f"处理后的数据集大小: {len(processed_dataset)}")
    if len(processed_dataset) > 0:
        logger.info(f"处理后的数据集样本: {processed_dataset[0]}")
    return processed_dataset

def custom_data_collator(tokenizer):
    def collate_fn(examples):
        input_texts = [example["input_text"] for example in examples]
        label_texts = [example["label_text"] for example in examples]

        input_encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        label_encodings = tokenizer(label_texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")

        batch_size = len(examples)
        input_ids = []
        attention_mask = []
        labels = []

        for i in range(batch_size):
            input_ids_i = input_encodings["input_ids"][i]
            label_ids_i = label_encodings["input_ids"][i]

            combined_ids = torch.cat([input_ids_i, label_ids_i])
            combined_mask = torch.cat([input_encodings["attention_mask"][i], label_encodings["attention_mask"][i]])

            if combined_ids.size(0) > MAX_SEQ_LENGTH:
                combined_ids = combined_ids[:MAX_SEQ_LENGTH]
                combined_mask = combined_mask[:MAX_SEQ_LENGTH]

            label_length = label_ids_i.size(0)
            input_length = input_ids_i.size(0)
            total_length = combined_ids.size(0)
            labels_i = torch.full_like(combined_ids, -100)
            if input_length < total_length:
                labels_i[-label_length:] = combined_ids[-label_length:]

            input_ids.append(combined_ids)
            attention_mask.append(combined_mask)
            labels.append(labels_i)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return collate_fn

def create_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # predictions 是 logits，形状 (batch_size, seq_len, vocab_size)
        # labels 是 token IDs，形状 (batch_size, seq_len)
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        # 计算交叉熵损失
        # 展平 predictions 和 labels 以便计算
        batch_size, seq_len, vocab_size = predictions.shape
        predictions = predictions.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        labels = labels.view(-1)  # (batch_size * seq_len)

        # 创建掩码，忽略 -100 和 pad_token_id
        valid_mask = (labels != -100) & (labels != tokenizer.pad_token_id)

        # 仅对有效 token 计算交叉熵
        valid_predictions = predictions[valid_mask]
        valid_labels = labels[valid_mask]

        if valid_labels.numel() == 0:  # 防止空标签
            cross_entropy = 0.0
        else:
            cross_entropy = torch.nn.functional.cross_entropy(
                valid_predictions, valid_labels, reduction="mean"
            ).item()

        # （可选）计算 token 预测准确率
        pred_tokens = torch.argmax(predictions, dim=-1)  # (batch_size * seq_len)
        correct_tokens = (pred_tokens == labels) & valid_mask
        total_valid_tokens = valid_mask.sum().item()
        token_accuracy = correct_tokens.sum().item() / total_valid_tokens if total_valid_tokens > 0 else 0.0

        return {
            "cross_entropy": cross_entropy,
            "token_accuracy": token_accuracy,
        }
    return compute_metrics

def train(model, tokenizer, train_dataset, eval_dataset):
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=30,
        save_steps=100,
        save_strategy="steps",
        save_total_limit=5,
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
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=custom_data_collator(tokenizer),
        compute_metrics=compute_metrics,  # 添加 compute_metrics
    )

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