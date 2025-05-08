import os
import torch
import logging
import time
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
DATASET_NAME = "Jofthomas/hermes-function-calling-thinking-V1"
OUTPUT_DIR = os.path.join(".", "fine_tuned_model")
MAX_SEQ_LENGTH = 4096
LORA_RANK = 32
LORA_ALPHA = 64
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 3

# Qwen chat template
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

            # 确保 model_type 和 vocab_size
            model.config.model_type = "qwen2"
            model.config.vocab_size = len(tokenizer)  # 设置初始 vocab_size

            # 添加特殊标记
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
            logger.info("Adding special tokens to tokenizer")
            num_added_tokens = tokenizer.add_special_tokens(special_tokens)
            logger.info(f"Added {num_added_tokens} special tokens: {special_tokens['additional_special_tokens']}")

            # 调整模型嵌入层并更新 config
            logger.info("Resizing model token embeddings")
            model.resize_token_embeddings(len(tokenizer))
            # model.config.vocab_size = len(tokenizer)  # 更新 vocab_size
            logger.info(f"Model token embeddings resized to {len(tokenizer)}")

            # 设置 Qwen chat template
            logger.info("Setting Qwen chat template")
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE

            # 应用 LoRA 适配器
            logger.info("Applying LoRA adapters")
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
            logger.info("LoRA adapters applied successfully")

            return model, tokenizer
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {str(e)}")
            if attempt == max_retries:
                raise
            time.sleep(retry_delay)

def prepare_dataset():
    """Load the dataset."""
    logger.info(f"Loading dataset {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split="train")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Dataset sample: {dataset[0]}")
    return dataset

def preprocess_dataset(dataset, tokenizer):
    """Preprocess dataset to create input-label pairs for each model response."""
    def format_conversation(examples):
        input_texts = []
        label_texts = []

        for example in examples["conversations"]:
            if not isinstance(example, list) or not example:
                logger.warning(f"Invalid conversation format in example: {example}")
                continue

            # Split multi-turn dialogue into input-label pairs
            for i in range(len(example)):
                if example[i]["role"] == "model":
                    # Input: all messages up to but not including the current model response
                    input_messages = example[:i]
                    # Label: the current model response
                    label_message = [example[i]]

                    # Format input and label using chat template
                    try:
                        input_text = tokenizer.apply_chat_template(input_messages, tokenize=False)
                        label_text = tokenizer.apply_chat_template(label_message, tokenize=False)
                        logger.debug(f"Input text: {input_text}")
                        logger.debug(f"Label text: {label_text}")
                        input_texts.append(input_text)
                        label_texts.append(label_text)
                    except Exception as e:
                        logger.error(f"Error applying chat_template: {str(e)}")
                        continue

        return {"input_text": input_texts, "label_text": label_texts}

    logger.info("Preprocessing dataset")
    # Map to create input-label pairs with batched processing
    processed_dataset = dataset.map(
        format_conversation,
        batched=True,
        remove_columns=["conversations"]
    )
    # Flatten the dataset to create one example per input-label pair
    processed_dataset = processed_dataset.flatten()
    # Filter out invalid examples
    processed_dataset = processed_dataset.filter(
        lambda x: x["input_text"] is not None and x["label_text"] is not None
    )
    logger.info(f"Processed dataset size: {len(processed_dataset)}")
    if len(processed_dataset) > 0:
        logger.info(f"Processed dataset sample: {processed_dataset[0]}")
    return processed_dataset

def custom_data_collator(tokenizer):
    """Custom data collator to mask input tokens and compute loss only on label tokens."""
    def collate_fn(examples):
        # Prepare lists for input and label texts
        input_texts = [example["input_text"] for example in examples]
        label_texts = [example["label_text"] for example in examples]

        # Tokenize inputs and labels separately
        input_encodings = tokenizer(input_texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        label_encodings = tokenizer(label_texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")

        # Create input_ids by concatenating input and label tokens
        batch_size = len(examples)
        input_ids = []
        attention_mask = []
        labels = []

        for i in range(batch_size):
            input_ids_i = input_encodings["input_ids"][i]
            label_ids_i = label_encodings["input_ids"][i]

            # Concatenate input and label tokens
            combined_ids = torch.cat([input_ids_i, label_ids_i])
            combined_mask = torch.cat([input_encodings["attention_mask"][i], label_encodings["attention_mask"][i]])

            # Truncate to MAX_SEQ_LENGTH if necessary
            if combined_ids.size(0) > MAX_SEQ_LENGTH:
                combined_ids = combined_ids[:MAX_SEQ_LENGTH]
                combined_mask = combined_mask[:MAX_SEQ_LENGTH]

            # Create labels: mask input tokens with -100, keep label tokens
            label_length = label_ids_i.size(0)
            input_length = input_ids_i.size(0)
            total_length = combined_ids.size(0)
            labels_i = torch.full_like(combined_ids, -100)  # Mask all tokens by default
            if input_length < total_length:
                # Set labels for the label_text portion (last label_length tokens)
                labels_i[-label_length:] = combined_ids[-label_length:]

            input_ids.append(combined_ids)
            attention_mask.append(combined_mask)
            labels.append(labels_i)

        # Pad sequences to the longest in the batch
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return collate_fn

def train(model, tokenizer, dataset):
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_steps=5,
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
        resume_from_checkpoint=True,
    )
        
    # 检查现有检查点
    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    resume_from_checkpoint = None
    if checkpoint_dirs:
        # 找到最新检查点
        latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("checkpoint-")[-1]))
        logger.info(f"Found existing checkpoint: {latest_checkpoint}")
        resume_from_checkpoint = latest_checkpoint
    else:
        logger.info("No existing checkpoints found, starting from scratch")        

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=custom_data_collator(tokenizer),
    )

    logger.info("Starting training")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        #trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    logger.info("Training completed")

    logger.info("Saving final model")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

def main():
    """Main function."""
    logger.info("Starting training process")
    model, tokenizer = load_model_and_tokenizer()
    dataset = prepare_dataset()
    processed_dataset = preprocess_dataset(dataset, tokenizer)
    train(model, tokenizer, processed_dataset)
    logger.info("Training process completed successfully")

if __name__ == "__main__":
    main()