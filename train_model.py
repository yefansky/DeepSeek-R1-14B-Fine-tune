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
MAX_SEQ_LENGTH = 4096  # 调整为适合多段对话的长度
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

            # 检查是否有预定义的 chat_template
            if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
                logger.info("No chat_template found, applying default ChatML format")
                from trl import setup_chat_format
                model, tokenizer = setup_chat_format(model, tokenizer)

            # 应用LoRA适配器
            logger.info("Applying LoRA adapters")
            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                modules_to_save=["lm_head", "embed_tokens"],  # 确保支持特殊标记
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
        bf16=True,
        save_strategy="steps",
        save_total_limit=2,
        optim="adamw_bnb_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        max_grad_norm=1.0,
        packing=True,  # 启用打包
        eval_packing=False,  # 验证集不打包
        max_seq_length=MAX_SEQ_LENGTH,
        eos_token="<|im_end|>",  # 设置 Qwen 的 EOS 标记
        neftune_noise_alpha=5,  # 添加 NEFTune 提升性能
        dataset_kwargs={"skip_prepare_dataset": True},  # 跳过默认数据集处理
        remove_unused_columns=False,  # 保留 messages 字段
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        packing=True,  # 确保与 SFTConfig 一致
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