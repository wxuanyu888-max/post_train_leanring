#!/usr/bin/env python3
"""
合并中英文训练数据
- 中文：sft-1000
- 英文：alpaca-1000
"""
import json
import random
from pathlib import Path


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print("=" * 60)
    print("合并中英文训练数据")
    print("=" * 60)

    # 加载中文数据
    zh_data = load_jsonl("./data/sft-1000.jsonl")
    print(f"\n中文数据 (sft-1000): {len(zh_data)} 条")

    # 加载英文数据
    en_data = load_jsonl("./data/processed/alpaca-1000.jsonl")
    print(f"英文数据 (alpaca-1000): {len(en_data)} 条")

    # 合并并打乱
    all_data = zh_data + en_data
    random.seed(42)
    random.shuffle(all_data)

    print(f"\n合并后总数：{len(all_data)} 条")
    print(f"中文占比：{len(zh_data)/len(all_data)*100:.1f}%")
    print(f"英文占比：{len(en_data)/len(all_data)*100:.1f}%")

    # 保存训练集
    train_path = "./data/processed/train.jsonl"
    save_jsonl(all_data, train_path)
    print(f"\n已保存至：{train_path}")

    # 验证集保持不变（还是 mini_benchmarks 180 条）
    print("\n验证集：./data/processed/valid.jsonl (180 条，保持不变)")

    print("\n" + "=" * 60)
    print("数据合并完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
