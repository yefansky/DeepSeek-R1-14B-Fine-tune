# train_model.py
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments
import logging
import json

# ������־
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ���ó���
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_NAME = "Trelis/function_calling_extended"
OUTPUT_DIR = ".\\fine_tuned_model"
MAX_SEQ_LENGTH = 32768
LORA_RANK = 16
LORA_ALPHA = 32
BATCH_SIZE = 2
LEARNING_RATE = 2e-5
EPOCHS = 3

def load_model_and_tokenizer():
    """����ģ�ͺ�tokenizer"""
    logger.info(f"Loading model {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def prepare_lora_config():
    """׼��LoRA����"""
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def prepare_dataset():
    """׼�����ݼ�"""
    logger.info(f"Loading dataset {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    def format_function(example):
        # �������������ݸ�ʽ���߼�
        return example
    
    dataset = dataset.map(format_function)
    return dataset

def train(model, tokenizer, dataset):
    """ѵ��ģ��"""
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
        save_total_limit=2
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=True
    )
    
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")
    
    logger.info("Saving model")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def main():
    """������"""
    logger.info("Starting training process")
    model, tokenizer = load_model_and_tokenizer()
    dataset = prepare_dataset()
    train(model, tokenizer, dataset)
    logger.info("Training process completed successfully")

if __name__ == "__main__":
    main()