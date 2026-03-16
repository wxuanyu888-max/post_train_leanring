#!/usr/bin/env python3
"""
准备 SFT-1000 训练数据
- 从 sft-1000.jsonl 划分训练集和验证集
- 将 mini_benchmarks (180 题) 转换为验证集格式
"""
import json
import random
from pathlib import Path


def load_jsonl(path):
    """加载 JSONL 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    """保存为 JSONL 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_mini_benchmarks_to_valid(data):
    """
    将 mini_benchmarks 转换为 SFT 验证集格式
    将 prompt 拆分为 instruction + input，answer 作为 output
    """
    converted = []
    for item in data:
        prompt = item.get("prompt", "")
        answer = item.get("answer", "")
        benchmark = item.get("benchmark", "unknown")
        qtype = item.get("type", "multiple_choice")

        # 根据题型转换格式
        if qtype == "multiple_choice":
            # 选择题格式
            instruction = f"回答以下{benchmark}选择题，只回答 A、B、C 或 D。"
            input_text = prompt
        elif qtype == "math_word_problem":
            # 数学题格式
            instruction = "解答以下数学应用题，请写出解题步骤，最后给出答案。"
            input_text = prompt
        elif qtype == "text_completion":
            # 文本补全格式
            instruction = "补全以下文本，使其逻辑通顺。"
            input_text = prompt
        else:
            # 通用格式
            instruction = f"回答以下问题（来源：{benchmark}）。"
            input_text = prompt

        converted.append({
            "instruction": instruction,
            "input": input_text,
            "output": answer,
            "source": f"mini_benchmarks_{benchmark}"
        })

    return converted


def main():
    print("=" * 60)
    print("准备 SFT-1000 训练数据")
    print("=" * 60)

    # 1. 加载 sft-1000 数据
    sft_1000_path = "./data/sft-1000.jsonl"
    print(f"\n加载 {sft_1000_path}...")
    sft_data = load_jsonl(sft_1000_path)
    print(f"  共 {len(sft_data)} 条数据")

    # 2. 打乱数据
    random.seed(42)
    random.shuffle(sft_data)

    # 3. 划分训练集（900 条）
    train_data = sft_data[:900]
    print(f"\n训练集：{len(train_data)} 条")

    # 4. 加载 mini_benchmarks 作为验证集
    mini_bench_path = "./bench_test_data/mini_benchmarks.jsonl"
    print(f"\n加载 {mini_bench_path}...")
    bench_data = load_jsonl(mini_bench_path)
    print(f"  共 {len(bench_data)} 条数据")

    # 5. 转换格式
    valid_data = convert_mini_benchmarks_to_valid(bench_data)
    print(f"\n验证集（转换后）：{len(valid_data)} 条")

    # 6. 保存数据
    train_path = "./data/processed/train.jsonl"
    valid_path = "./data/processed/valid.jsonl"

    save_jsonl(train_data, train_path)
    print(f"\n保存训练集至：{train_path}")

    save_jsonl(valid_data, valid_path)
    print(f"保存验证集至：{valid_path}")

    # 7. 打印统计
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"  训练集：{len(train_data)} 条 (来自 sft-1000)")
    print(f"  验证集：{len(valid_data)} 条 (来自 mini_benchmarks 180 题)")
    print("=" * 60)
    print("\n数据准备完成！")


if __name__ == "__main__":
    main()
