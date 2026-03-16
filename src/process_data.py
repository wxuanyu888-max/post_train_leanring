#!/usr/bin/env python3
"""
数据清洗脚本
将原始训练数据清洗后保存到 data/processed/train.jsonl
"""

import argparse
import json
import random
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/data_config.yaml") -> dict:
    """加载数据配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: str) -> list[dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"跳过无效 JSON 行: {line[:50]}...")
    return data


def save_jsonl(data: list[dict], file_path: str):
    """保存为 JSONL 文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已保存 {len(data)} 条数据到 {file_path}")


def clean_data(data: list[dict],
               min_instruction_len: int = 3,
               min_output_len: int = 1,
               remove_duplicates: bool = True,
               shuffle: bool = True,
               seed: int = 42) -> list[dict]:
    """清洗数据"""
    print(f"原始数据量: {len(data)} 条")

    # 1. 过滤无效数据
    cleaned = []
    for item in data:
        # 检查必要字段
        if "instruction" not in item or "output" not in item:
            continue

        instruction = item.get("instruction", "").strip()
        output = item.get("output", "").strip()
        input_text = item.get("input", "").strip()

        # 过滤空值和过短内容
        if len(instruction) < min_instruction_len:
            continue
        if len(output) < min_output_len:
            continue

        cleaned.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })

    print(f"过滤无效数据后: {len(cleaned)} 条")

    # 2. 去重
    if remove_duplicates:
        seen = set()
        unique_data = []
        for item in cleaned:
            # 用 instruction + output 作为唯一标识
            key = (item["instruction"], item["output"])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        cleaned = unique_data
        print(f"去重后: {len(cleaned)} 条")

    # 3. 随机打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(cleaned)
        print("已打乱数据")

    return cleaned


def main():
    parser = argparse.ArgumentParser(description="数据清洗脚本")
    parser.add_argument("--config", type=str, default="configs/data_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--input", type=str, default=None,
                        help="输入文件路径（覆盖配置）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认: data/processed/train.jsonl）")
    parser.add_argument("--min-instruction-len", type=int, default=3,
                        help="指令最小长度")
    parser.add_argument("--min-output-len", type=int, default=1,
                        help="输出最小长度")
    parser.add_argument("--no-deduplicate", action="store_true",
                        help="不去重")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 获取输入输出路径
    if args.input:
        input_path = args.input
    else:
        input_path = config["data"]["train_data"]

    if args.output:
        output_path = args.output
    else:
        output_path = f"{config['data']['output_dir']}/train.jsonl"

    print(f"\n=== 数据清洗 ===")
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")

    # 加载数据
    print("\n加载数据...")
    data = load_jsonl(input_path)

    # 清洗数据
    print("\n清洗数据...")
    cleaned_data = clean_data(
        data,
        min_instruction_len=args.min_instruction_len,
        min_output_len=args.min_output_len,
        remove_duplicates=not args.no_deduplicate,
        seed=args.seed
    )

    # 保存结果
    print("\n保存结果...")
    save_jsonl(cleaned_data, output_path)

    print("\n完成!")


if __name__ == "__main__":
    main()
