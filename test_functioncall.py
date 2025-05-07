import json
import os
import shutil
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def create_temp_model_dir(original_path, temp_path):
    """复制模型目录到临时目录并修改 tokenizer_config.json"""
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    shutil.copytree(original_path, temp_path)
    
    # 修改临时目录中的 tokenizer_config.json
    tokenizer_config_path = os.path.join(temp_path, 'tokenizer_config.json')
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 设置与训练一致的 chat_template，添加 add_generation_prompt
        config['chat_template'] = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'human' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'model' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'tool' %}<|im_start|>tool\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}"
        )
        
        with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    else:
        raise FileNotFoundError("未找到 tokenizer_config.json")

def load_model_and_tokenizer(model_path):
    # 加载模型和 tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=32768,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )
    
    # 调试：确认 chat_template
    print("Tokenizer chat_template:", tokenizer.chat_template)
    
    return model, tokenizer

def generate_response(model, tokenizer, messages):
    # 使用 apply_chat_template 格式化多轮对话
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
    """提取工具调用内容并进行验证"""
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)
    
    if start_idx == -1 or end_idx == -1:
        raise ValueError("未检测到工具调用标记")
    
    json_str = response[start_idx+len(start_tag):end_idx].strip()
    try:
        tool_call = json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError("工具调用内容不是有效的JSON格式")
    
    # 验证必需字段
    required_fields = ["name", "arguments"]
    for field in required_fields:
        if field not in tool_call:
            raise ValueError(f"缺少必需字段: {field}")
    
    return tool_call

def test_tool_calling(model_path):
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # 工具定义示例
    tools = [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期（YYYY-MM-DD格式）",
                        "default": "今天"
                    }
                },
                "required": ["city"]
            }
        }
    ]
    
    # 系统提示，指导模型进行工具调用
    system_prompt = f"""你是一个支持工具调用的助手。以下是可用工具的定义：
{json.dumps(tools, indent=2, ensure_ascii=False)}

请根据用户输入选择合适的工具，并以以下格式返回工具调用：
<tool_call>{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}</tool_call>

如果无需工具调用，直接返回答案。"""
    
    # 测试案例（模拟多轮对话）
    test_cases = [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": "请问杭州明天的天气怎么样？"}
            ],
            "expected": {
                "name": "get_weather",
                "arguments": {
                    "city": "杭州",
                    "date": "明天"
                }
            }
        },
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "human", "content": "帮我查下纽约的天气情况"},
                {"role": "model", "content": "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"纽约\"}}</tool_call>"},
                {"role": "human", "content": "好的，那后天的天气呢？"}
            ],
            "expected": {
                "name": "get_weather",
                "arguments": {
                    "city": "纽约",
                    "date": "后天"
                }
            }
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n=== 测试案例 {i+1} ===")
        
        # 生成响应
        response = generate_response(model, tokenizer, case["messages"])
        print("\n完整模型输出：")
        print(response)
        
        try:
            # 提取并验证工具调用
            tool_call = extract_tool_call(response)
            
            # 验证结果
            assert tool_call["name"] == case["expected"]["name"], "工具名称不匹配"
            
            # 验证参数
            for param, value in case["expected"]["arguments"].items():
                assert param in tool_call["arguments"], f"缺少参数: {param}"
                assert tool_call["arguments"][param] == value, f"参数值不匹配: {param}"
            
            print("\n✅ 测试通过")
            print("提取到的工具调用：")
            print(json.dumps(tool_call, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"\n❌ 测试失败：{str(e)}")

if __name__ == "__main__":
    original_model_path = "./fine_tuned_model/checkpoint-100"  # 原始模型路径
    temp_model_path = "./temp_model"  # 临时目录
    
    # 创建临时模型目录
    create_temp_model_dir(original_model_path, temp_model_path)
    
    # 使用临时目录运行测试
    test_tool_calling(temp_model_path)