import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
import logging
import json
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_NAME = "Jofthomas/hermes-function-calling-thinking-V1"
OUTPUT_DIR = ".\\fine_tuned_model"
MAX_SEQ_LENGTH = 32768
LORA_RANK = 32
LORA_ALPHA = 64
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
EPOCHS = 3

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

            # 应用LoRA适配器
            logger.info("Applying LoRA adapters")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing=True,
                random_state=3407,
                max_seq_length=MAX_SEQ_LENGTH
            )
            logger.info("LoRA adapters applied successfully")

            # 打印可训练参数
            logger.info("Trainable parameters:")
            trainable_params = 0
            total_params = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    logger.info(f" - {name}: {param.shape}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")

            return model, tokenizer
        except OSError as e:
            logger.warning(f"Attempt {attempt} failed with OSError: {str(e)}")
            if attempt == max_retries:
                logger.error(f"All {max_retries} attempts failed. Final error: {str(e)}")
                logger.error("Please check the model repository at: "
                             "https://huggingface.co/unsloth/deepseek-r1-distill-qwen-14b-unsloth-bnb-4bit")
                raise
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error during attempt {attempt}: {str(e)}")
            raise
    raise Exception("Failed to load model after retries")

def prepare_lora_config():
    """准备LoRA配置"""
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def prepare_dataset():
    """准备数据集"""
    logger.info(f"Loading dataset {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    def format_conversation(example):
        """处理单条对话样本"""
        formatted_text = ""
        for msg in example["conversations"]:  # 直接访问当前样本的conversations字段
            if msg["role"] == "system":
                formatted_text += f"<system>\n{msg['content']}\n</system>\n"
            elif msg["role"] == "human":
                formatted_text += f"<user>\n{msg['content']}\n</user>\n"
            elif msg["role"] == "model":
                formatted_text += f"<assistant>\n{msg['content']}\n</assistant>\n"
            elif msg["role"] == "tool":
                formatted_text += f"{msg['content']}\n"
        return {"text": formatted_text}
    
    # 打印原始样本
    logger.info(f"原始样本结构:\n{dataset[0]}")
    
    # 先处理数据集再获取样本
    formatted_dataset = dataset.map(format_conversation)
    
    # 打印处理后的样本
    logger.info(f"\n处理后的样本:\n{formatted_dataset[0]['text']}")
    
    return formatted_dataset

def train(model, tokenizer, dataset):
    """训练模型"""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_steps=100,
        fp16=False,
        bf16=True,
        save_strategy="steps",
        save_total_limit=2,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
    )
    
    def formatting_func(example):
        # 直接返回已经处理好的文本
        return {"text": example["text"]}
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=True,
        formatting_func=formatting_func,
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
    train(model, tokenizer, dataset)
    logger.info("Training process completed successfully")

if __name__ == "__main__":
    main()