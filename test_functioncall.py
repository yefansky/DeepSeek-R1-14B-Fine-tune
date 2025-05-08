import json
import os
import glob
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import logging
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    # Load the checkpoint’s tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get the vocabulary size from the tokenizer
    vocab_size = len(tokenizer)  # Dynamically retrieve vocab size

    # Load the base model
    base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=32768,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )

    # Resize the base model’s embeddings to match the checkpoint’s vocab size
    model.resize_token_embeddings(vocab_size)

    # Load the LoRA adapters
    model = PeftModel.from_pretrained(
        model,
        model_path,
        is_trainable=False
    )

    return model, tokenizer

def generate_response(model, tokenizer, messages):
    """Generate response using chat template"""
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_tool_call(response):
    """Extract and validate tool call"""
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("No tool call tags detected")
    
    json_str = response[start_idx+len(start_tag):end_idx].strip()
    try:
        tool_call = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("Tool call content is not valid JSON")
    
    required_fields = ["name", "arguments"]
    for field in required_fields:
        if field not in tool_call:
            raise ValueError(f"Missing required field: {field}")
    
    return tool_call

def test_tool_calling(model_path):
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Tool definition
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "date": {"type": "string", "description": "Query date (YYYY-MM-DD)", "default": "today"}
                },
                "required": ["city"]
            }
        }
    ]
    
    # System prompt
    system_prompt = f"""You are an assistant that supports tool calling. Available tools:
{json.dumps(tools, indent=2, ensure_ascii=False)}

Respond with tool calls in this format:
<tool_call>{{"name": "tool_name", "arguments": {{"param": "value"}}}}</tool_call>

If no tool is needed, return the answer directly."""
    
    # Test cases
    test_cases = [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": "What's the weather in Hangzhou tomorrow?"}
            ],
            "expected": {
                "name": "get_weather",
                "arguments": {"city": "杭州", "date": "明天"}
            }
        },
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": "Check New York's weather"},
                {"role": "model", "content": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"纽约\"}}</tool_call>"},
                {"role": "human", "content": "What about the day after tomorrow?"}
            ],
            "expected": {
                "name": "get_weather",
                "arguments": {"city": "纽约", "date": "后天"}
            }
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1} ===")
        
        # Generate response
        response = generate_response(model, tokenizer, case["messages"])
        print("\nModel Output:")
        print(response)
        
        try:
            # Extract and validate tool call
            tool_call = extract_tool_call(response)
            
            # Verify results
            assert tool_call["name"] == case["expected"]["name"], "Tool name mismatch"
            for param, value in case["expected"]["arguments"].items():
                assert param in tool_call["arguments"], f"Missing parameter: {param}"
                assert tool_call["arguments"][param] == value, f"Parameter value mismatch: {param}"
            
            print("\n✅ Test Passed")
            print("Extracted Tool Call:")
            print(json.dumps(tool_call, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"\n❌ Test Failed: {str(e)}")

if __name__ == "__main__":
    OUTPUT_DIR = "./fine_tuned_model"  # Base directory for checkpoints
    
    # Find the latest checkpoint
    checkpoint_dirs = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
    if checkpoint_dirs:
        model_path = max(checkpoint_dirs, key=lambda x: int(x.split("checkpoint-")[-1]))
        logger.info(f"Found latest checkpoint: {model_path}")
    else:
        logger.error("No checkpoints found in {OUTPUT_DIR}")
        raise FileNotFoundError(f"No checkpoints found in {OUTPUT_DIR}")
    
    test_tool_calling(model_path)