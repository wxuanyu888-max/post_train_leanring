#!/usr/bin/env python3
"""
基座模型效果测试脚本
测试 Qwen2.5-0.5B-Instruct 在微调前的原始效果
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import json


def load_model_and_tokenizer(model_path: str):
    """加载模型和分词器"""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    model.eval()
    print(f"Model loaded successfully!")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """生成响应"""
    # 构建提示
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

    # 分词
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response


def test_single_instruction(
    model,
    tokenizer,
    instruction: str,
    input_text: str = "",
    reference: str = "",
):
    """测试单个指令"""
    print("\n" + "=" * 70)
    print(f"Instruction: {instruction}")
    if input_text:
        print(f"Input: {input_text}")
    if reference:
        print(f"Reference: {reference}")
    print("-" * 70)

    response = generate_response(model, tokenizer, instruction, input_text)
    print(f"Response: {response}")

    return response


def run_demo_tests(model, tokenizer):
    """运行演示测试"""
    print("\n" + "=" * 70)
    print("BASE MODEL DEMO TESTS")
    print("=" * 70)

    test_cases = [
        # 知识问答
        {
            "instruction": "What is machine learning?",
            "input": "",
            "reference": "Machine learning is a subset of artificial intelligence...",
        },
        # 数学计算
        {
            "instruction": "Solve this math problem: 15 + 27 * 3 =",
            "input": "",
            "reference": "Following order of operations: 27 * 3 = 81, then 15 + 81 = 96",
        },
        # 文本创作
        {
            "instruction": "Write a short poem about spring",
            "input": "",
            "reference": "",
        },
        # 代码生成
        {
            "instruction": "Write a Python function to calculate factorial",
            "input": "",
            "reference": "def factorial(n): ...",
        },
        # 翻译
        {
            "instruction": "Translate the following sentence to Chinese",
            "input": "Hello, how are you today?",
            "reference": "你好，你今天好吗？",
        },
        # 逻辑推理
        {
            "instruction": "If all roses are flowers, and some flowers fade quickly, do all roses fade quickly?",
            "input": "",
            "reference": "No, we cannot conclude that all roses fade quickly...",
        },
        # 摘要
        {
            "instruction": "Summarize the following text in one sentence",
            "input": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet and is commonly used for typing practice.",
            "reference": "",
        },
    ]

    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n>>> Test {i}/{len(test_cases)}")
        response = test_single_instruction(
            model,
            tokenizer,
            case["instruction"],
            case["input"],
            case.get("reference", ""),
        )
        results.append({
            "instruction": case["instruction"],
            "input": case["input"],
            "reference": case.get("reference", ""),
            "generated": response,
        })

    return results


def run_data_comparison(model, tokenizer, data_path: str = "./data/raw/alpaca-500.jsonl", max_samples: int = 10):
    """在真实数据上运行对比测试"""
    print("\n" + "=" * 70)
    print(f"DATA COMPARISON TEST (using {data_path})")
    print("=" * 70)

    # 加载测试数据
    test_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            test_data.append(json.loads(line))

    print(f"Loaded {len(test_data)} samples for testing")

    results = []
    for i, sample in enumerate(test_data, 1):
        print(f"\n\n>>> Sample {i}/{len(test_data)}")

        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        reference = sample.get("output", "")

        response = generate_response(model, tokenizer, instruction, input_text)

        print(f"Instruction: {instruction[:50]}...")
        if input_text:
            print(f"Input: {input_text[:50]}...")
        print(f"Reference: {reference[:100]}...")
        print("-" * 50)
        print(f"Generated: {response[:200]}...")

        results.append({
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "generated": response,
        })

    return results


def interactive_mode(model, tokenizer):
    """交互式测试模式"""
    print("\n" + "=" * 70)
    print("INTERACTIVE TEST MODE")
    print("Type 'quit' or 'exit' to exit")
    print("=" * 70)

    while True:
        instruction = input("\n[Instruction] > ").strip()

        if instruction.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        input_text = input("[Input] (optional, press Enter to skip) > ").strip()

        response = generate_response(
            model,
            tokenizer,
            instruction,
            input_text if input_text else "",
            max_new_tokens=256,
            temperature=0.7,
        )

        print(f"\n[Response]\n{response}")


def save_results(results: List[Dict], output_path: str):
    """保存结果"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test base model performance")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/qwen2.5-0.5b-instruct",
        help="Path to the base model",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "data", "interactive"],
        default="demo",
        help="Test mode: demo, data, or interactive",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/raw/alpaca-500.jsonl",
        help="Path to test data (for data mode)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to test (for data mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/base_model_results.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )

    args = parser.parse_args()

    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("CUDA not available, using CPU")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    # 根据模式运行
    if args.mode == "demo":
        results = run_demo_tests(model, tokenizer)
    elif args.mode == "data":
        results = run_data_comparison(
            model,
            tokenizer,
            args.data_path,
            args.max_samples,
        )
    elif args.mode == "interactive":
        interactive_mode(model, tokenizer)
        return

    # 保存结果
    save_results(results, args.output)

    # 打印统计
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total samples tested: {len(results)}")
    print(f"Results saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
