import json
import os
import glob
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import logging
from peft import PeftModel
from unsloth.chat_templates import get_chat_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,#base_model_name,
        max_seq_length=32768,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen-2.5",
    )

    return model, tokenizer

def generate_response(model, tokenizer, messages):
    # 步骤 1: 应用聊天模板，生成格式化字符串
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 输出字符串
        add_generation_prompt=True  # 添加生成提示
    )
    
    # 步骤 2: 手动 token 化
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,  # 与训练时的 MAX_SEQ_LENGTH 一致
        padding=True
    ).to("cuda")
    
    input_length = inputs["input_ids"].shape[1]
    
    # 步骤 3: 生成回复
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    output_length = outputs.shape[1]
    
    # 步骤 4: 截断输入部分，仅解码新生成内容
    if output_length <= input_length:
        print("错误：未生成新内容！")
        print("完整输出:", tokenizer.decode(outputs[0], skip_special_tokens=False))
        return ""
    
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text

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

    test_text = "<tools>test<tool_call></tool_call></tools>"
    test_encodings = tokenizer(
        test_text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    logger.info("分词测试结果:")
    logger.info(f"输入文本: {test_text}")
    logger.info(f"Tokenized IDs: {test_encodings['input_ids']}")
    logger.info(f"解码结果: {tokenizer.decode(test_encodings['input_ids'])}")
    logger.info(f"偏移映射: {test_encodings['offset_mapping']}")
    
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
                {"role": "user", "content": "What's the weather in Hangzhou tomorrow?"}
            ],
            "expected": {
                "name": "get_weather",
                "arguments": {"city": "杭州", "date": "明天"}
            }
        },
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Check New York's weather"},
                {"role": "assistant", "content": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"纽约\"}}</tool_call>"},
                {"role": "user", "content": "What about the day after tomorrow?"}
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
    
    #model_path = os.path.join(OUTPUT_DIR, "checkpoint-50")
    test_tool_calling(model_path)