import json  # 添加这行导入
import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def load_model_and_tokenizer(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 32768,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(
        **inputs,
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
    
    # 测试案例
    test_cases = [
        {
            "instruction": "你需要使用提供的工具来回答问题",
            "input": "请问杭州明天的天气怎么样？",
            "expected": {
                "name": "get_weather",
                "arguments": {
                    "city": "杭州",
                    "date": "明天"
                }
            }
        },
        {
            "instruction": "请使用合适的工具进行查询",
            "input": "帮我查下纽约的天气情况",
            "expected": {
                "name": "get_weather",
                "arguments": {
                    "city": "纽约"
                }
            }
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n=== 测试案例 {i+1} ===")
        
        # 构建prompt
        prompt = f"""Instruction: {case['instruction']}
可用的工具定义：
{json.dumps(tools, indent=2, ensure_ascii=False)}

Input: {case['input']}
Output:"""
        
        # 生成响应
        response = generate_response(model, tokenizer, prompt)
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
    model_path = "./fine_tuned_model"  # 修改为你的模型路径
    test_tool_calling(model_path)