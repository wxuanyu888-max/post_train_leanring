#!/usr/bin/env python3
"""
快速效果测试脚本 - 非交互式
直接运行即可看到基座模型的测试效果
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_header(text: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f" {text} ")
    print("=" * 70 + "\n")


def load_model(model_path: str):
    """加载模型"""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate(model, tokenizer, instruction: str, input_text: str = "", max_tokens: int = 200):
    """生成响应"""
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return response


def run_tests():
    """运行测试"""
    print_header("Qwen2.5-0.5B 基座模型效果测试")

    model_path = "./models/qwen2.5-0.5b-instruct"

    # 测试用例
    test_cases = [
        {
            "category": "知识问答",
            "instruction": "What is artificial intelligence?",
            "input": "",
        },
        {
            "category": "数学计算",
            "instruction": "Calculate: 12 + 8 * 2 =",
            "input": "",
        },
        {
            "category": "代码生成",
            "instruction": "Write a Python function to add two numbers",
            "input": "",
        },
        {
            "category": "翻译",
            "instruction": "Translate to Chinese:",
            "input": "Good morning! How can I help you today?",
        },
        {
            "category": "文本创作",
            "instruction": "Write a two-sentence story about a robot",
            "input": "",
        },
    ]

    # 加载模型
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nPlease make sure the model files exist in:")
        print(f"  {model_path}")
        return

    # 运行测试
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}: {case['category']}")
        print(f"{'='*60}")
        print(f"Instruction: {case['instruction']}")
        if case['input']:
            print(f"Input: {case['input']}")
        print("-" * 60)

        result = generate(model, tokenizer, case['instruction'], case['input'], max_tokens=150)

        print(f"Response:\n{result}")

    print_header("测试完成")

    print("\n下一步:")
    print("  1. 数据预处理：python scripts/preprocess_data.py")
    print("  2. 开始训练：python scripts/train.py")
    print("  3. 对比效果：python scripts/compare_models.py")
    print()


if __name__ == "__main__":
    run_tests()
