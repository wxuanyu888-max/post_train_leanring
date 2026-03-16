#!/usr/bin/env python3
"""
基座模型测试脚本
测试未经过微调的 Qwen2.5-0.5B-Instruct 模型效果
"""
import torch
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    """加载模型"""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
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


def generate(model, tokenizer, instruction: str, input_text: str = "", max_tokens: int = 256):
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


def run_base_model_test(output_path: str = "./outputs/base_model_test_results.json"):
    """运行基座模型测试"""
    print("=" * 70)
    print(" Qwen2.5-0.5B-Instruct 基座模型测试 ")
    print("=" * 70)
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    model_path = "./models/qwen2.5-0.5b-instruct"

    # 综合测试用例
    test_cases = [
        # === 知识问答 ===
        {
            "category": "Knowledge",
            "instruction": "Explain what is machine learning in simple terms.",
            "input": "",
            "reference": "",
        },
        {
            "category": "Knowledge",
            "instruction": "What is the capital of France?",
            "input": "",
            "reference": "Paris",
        },

        # === 数学计算 ===
        {
            "category": "Math",
            "instruction": "Calculate: 25 + 17 * 3 =",
            "input": "",
            "reference": "76",
        },

        # === 代码生成 ===
        {
            "category": "Code",
            "instruction": "Write a Python function that checks if a number is prime.",
            "input": "",
            "reference": "",
        },

        # === 翻译 ===
        {
            "category": "Translation",
            "instruction": "Translate the following sentence to Chinese:",
            "input": "The weather is beautiful today.",
            "reference": "",
        },
        {
            "category": "Translation",
            "instruction": "Translate to English:",
            "input": "你好，很高兴认识你。",
            "reference": "",
        },

        # === 文本创作 ===
        {
            "category": "Creative",
            "instruction": "Write a haiku about nature.",
            "input": "",
            "reference": "",
        },

        # === 指令遵循 ===
        {
            "category": "Instruction Following",
            "instruction": "List 3 fruits in alphabetical order.",
            "input": "",
            "reference": "",
        },

        # === 逻辑推理 ===
        {
            "category": "Reasoning",
            "instruction": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "input": "",
            "reference": "5 minutes",
        },

        # === 客服场景 ===
        {
            "category": "Customer Service",
            "instruction": "Respond to a customer who is upset about a delayed shipment.",
            "input": "My order was supposed to arrive 3 days ago but it still hasn't come!",
            "reference": "",
        },
    ]

    # 加载模型
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        return None

    # 运行测试
    results = []
    correct_count = 0

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(test_cases)}] {case['category']}")
        print(f"{'='*60}")
        print(f"Instruction: {case['instruction']}")
        if case['input']:
            print(f"Input: {case['input']}")
        print("-" * 60)

        response = generate(model, tokenizer, case['instruction'], case['input'])

        print(f"Response:\n{response[:300]}..." if len(response) > 300 else f"Response:\n{response}")

        results.append({
            "id": i,
            "category": case["category"],
            "instruction": case["instruction"],
            "input": case["input"],
            "reference": case.get("reference", ""),
            "generated": response,
        })

    # 保存结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": model_path,
        "test_time": datetime.now().isoformat(),
        "total_tests": len(results),
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f" 测试完成！结果已保存到：{output_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_base_model_test()
