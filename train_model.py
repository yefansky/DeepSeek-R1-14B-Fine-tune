import os
import torch
import logging
import time
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForLanguageModeling

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
)

def load_model_and_tokenizer(max_retries=3, retry_delay=5):
    """Load model and tokenizer with LoRA adapters and add special tokens."""
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

            # Check tokenizer configuration
            logger.info(f"Tokenizer EOS token: {tokenizer.eos_token}")
            logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")
            logger.info(f"Tokenizer chat template: {getattr(tokenizer, 'chat_template', None)}")

            # Ensure correct EOS token
            if tokenizer.eos_token is None or tokenizer.eos_token != "<|im_end|>":
                logger.warning("EOS token is missing or incorrect, setting to <|im_end|>")
                tokenizer.eos_token = "<|im_end|>"

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                logger.warning("Pad token is missing, setting to EOS token (<|im_end|>)")
                tokenizer.pad_token = tokenizer.eos_token

            # Add special tokens
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

            # Verify special tokens
            for token in special_tokens["additional_special_tokens"]:
                token_id = tokenizer.convert_tokens_to_ids(token)
                logger.info(f"Token {token} assigned ID {token_id}")
                if token_id == tokenizer.unk_token_id:
                    logger.error(f"Token {token} was not properly added to the vocabulary")
                    raise ValueError(f"Token {token} is treated as unknown")

            # Resize model embedding layer to accommodate new tokens
            logger.info("Resizing model token embeddings")
            model.resize_token_embeddings(len(tokenizer))
            logger.info(f"Model token embeddings resized to {len(tokenizer)}")

            # Set Qwen chat template
            logger.info("Setting Qwen chat template")
            tokenizer.chat_template = QWEN_CHAT_TEMPLATE

            # Apply LoRA adapters
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

            # Log parameter counts
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
    """Train the model."""
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
        packing=False,
        max_seq_length=MAX_SEQ_LENGTH,
        neftune_noise_alpha=5,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=custom_data_collator(tokenizer),
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

    logger.info("Saving model")
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