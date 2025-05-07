# -*- coding: utf-8 -*-
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_NAME = "Jofthomas/hermes-function-calling-thinking-V1"
OUTPUT_DIR = os.path.join(".", "fine_tuned_model")  # 跨平台路径
MAX_SEQ_LENGTH = 4096  # 适合多段对话，RTX 5090 可支持
LORA_RANK = 32
LORA_ALPHA = 64
BATCH_SIZE = 4  # 增加 batch size，利用 RTX 5090 的 VRAM
LEARNING_RATE = 2e-5
EPOCHS = 3

# Qwen 的默认 chat_template
QWen_CHAT_TEMPLATE = (
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
)

def load_model_and_tokenizer(max_retries=300, retry_delay=5):
    """加载模型和tokenizer，并应用LoRA适配器"""
    logger.info(f"Loading model {MODEL_NAME}")
    for attempt in range(1, max_retries + 1):
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_NAME,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                resume_download=True
            )
            logger.info("Model and tokenizer loaded successfully")

            # 检查 tokenizer 配置
            logger.info(f"Tokenizer EOS token: {tokenizer.eos_token}")
            logger.info(f"Tokenizer chat template: {getattr(tokenizer, 'chat_template', None)}")
            
            # 确保 eos_token 正确（Qwen 预期为 <|im_end|>）
            if tokenizer.eos_token is None or tokenizer.eos_token != "<|im_end|>":
                logger.warning("EOS token is missing or incorrect, setting to <|im_end|>")
                tokenizer.eos_token = "<|im_end|>"

            # 设置 Qwen 的 chat_template
            logger.info("Setting Qwen chat_template")
            tokenizer.chat_template = QWen_CHAT_TEMPLATE

            # 应用LoRA适配器
            logger.info("Applying LoRA adapters")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                modules_to_save=["lm_head", "embed_tokens"],  # 支持特殊标记
                use_gradient_checkpointing=True,
                random_state=3407,
                max_seq_length=MAX_SEQ_LENGTH
            )
            logger.info("LoRA adapters applied successfully")

            # 打印可训练参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

            return model, tokenizer
        except OSError as e:
            logger.warning(f"Attempt {attempt} failed with OSError: {str(e)}")
            if attempt == max_retries:
                logger.error(f"All {max_retries} attempts failed. Final error: {str(e)}")
                raise
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error during attempt {attempt}: {str(e)}")
            raise
    raise Exception("Failed to load model after retries")

def prepare_dataset():
    """加载数据集"""
    logger.info(f"Loading dataset {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Dataset sample: {dataset[0]}")
    return dataset

def preprocess_dataset(dataset, tokenizer):
    """预处理数据集，将 conversations 转换为 text 字段"""
    def format_conversation(example):
        if "conversations" not in example:
            logger.error("Example missing 'conversations' field")
            return {"text": None}
        
        conversation = example["conversations"]
        if not isinstance(conversation, list) or not conversation:
            logger.warning(f"Invalid conversation format in example: {example}")
            return {"text": None}
        
        # 转换为 messages 格式
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in conversation]
        
        # 调试：记录 messages
        logger.debug(f"Messages for example: {messages}")
        
        # 使用 tokenizer 的 chat_template 格式化
        try:
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
            logger.debug(f"Formatted text: {formatted_text}")
            return {"text": formatted_text}
        except Exception as e:
            logger.error(f"Error applying chat_template: {str(e)}")
            return {"text": None}

    logger.info("Preprocessing dataset")
    processed_dataset = dataset.map(format_conversation, remove_columns=["conversations"])
    processed_dataset = processed_dataset.filter(lambda x: x["text"] is not None)
    logger.info(f"Processed dataset size: {len(processed_dataset)}")
    logger.info(f"Processed dataset sample: {processed_dataset[0]}")
    return processed_dataset

def train(model, tokenizer, dataset):
    """训练模型"""
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        bf16=True,  # RTX 5090 支持 bf16
        save_strategy="steps",
        save_total_limit=2,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        packing=False,  # 禁用打包以简化调试
        max_seq_length=MAX_SEQ_LENGTH,
        neftune_noise_alpha=5,  # 提升对话性能
        dataset_text_field="text",  # 指定 text 字段
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",  # 使用预处理的 text 字段
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

    logger.info("Saving model")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def main():
    """主函数"""
    logger.info("Starting training process")
    model, tokenizer = load_model_and_tokenizer()
    dataset = prepare_dataset()
    processed_dataset = preprocess_dataset(dataset, tokenizer)
    train(model, tokenizer, processed_dataset)
    logger.info("Training process completed successfully")

if __name__ == "__main__":
    main()