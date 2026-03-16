#!/usr/bin/env python3
"""
数据预处理脚本
将原始 JSONL 数据转换为训练可用格式
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess SFT data")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/raw",
        help="原始数据目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="处理后数据输出目录",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="最小样本长度过滤",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数限制",
    )

    return parser.parse_args()


def load_jsonl_file(file_path: Path) -> List[Dict]:
    """加载单个 JSONL 文件"""
    data = []
    errors = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  Warning: Skipping invalid JSON at line {i} in {file_path.name}")
    if errors > 0:
        print(f"  Skipped {errors} invalid lines in {file_path.name}")
    return data


def load_all_data(input_dir: Path) -> List[Dict]:
    """加载目录下所有 JSONL 文件"""
    all_data = []
    jsonl_files = list(input_dir.glob("*.jsonl"))

    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    for file_path in jsonl_files:
        data = load_jsonl_file(file_path)
        print(f"  Loaded {len(data)} samples from {file_path.name}")
        all_data.extend(data)

    return all_data


def filter_data(
    data: List[Dict],
    min_length: int = 10,
    max_samples: int = None,
) -> List[Dict]:
    """
    过滤数据

    Args:
        data: 原始数据
        min_length: 最小 output 长度
        max_samples: 最大样本数
    """
    filtered = []

    for item in data:
        output = item.get("output", "")
        instruction = item.get("instruction", "")

        # 过滤太短的样本
        if len(output) < min_length:
            continue

        # 过滤空 instruction
        if not instruction:
            continue

        filtered.append(item)

    print(f"Filtered from {len(data)} to {len(filtered)} samples")

    # 限制最大样本数
    if max_samples and max_samples > 0:
        filtered = filtered[:max_samples]
        print(f"Limited to {max_samples} samples")

    return filtered


def save_jsonl(data: List[Dict], output_path: Path) -> None:
    """保存为 JSONL 格式"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(data)} samples to {output_path}")


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 50)
    print("Data Preprocessing")
    print("=" * 50)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Test split: {args.test_split}")
    print(f"Min length: {args.min_length}")
    print("=" * 50)

    # 检查输入目录
    if not input_dir.exists():
        print(f"Warning: Input directory does not exist: {input_dir}")
        print("Creating it and copying sample data...")
        input_dir.mkdir(parents=True, exist_ok=True)
        return

    # 加载所有数据
    print("\nLoading data...")
    all_data = load_all_data(input_dir)

    if not all_data:
        print("No data found! Please add JSONL files to the input directory.")
        return

    print(f"Total samples loaded: {len(all_data)}")

    # 过滤数据
    print("\nFiltering data...")
    filtered_data = filter_data(
        all_data,
        min_length=args.min_length,
        max_samples=args.max_samples,
    )

    # 划分训练集和验证集
    print("\nSplitting data...")
    train_data, valid_data = train_test_split(
        filtered_data,
        test_size=args.test_split,
        random_state=args.seed,
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Valid samples: {len(valid_data)}")

    # 保存处理后的数据
    print("\nSaving processed data...")
    save_jsonl(train_data, output_dir / "train.jsonl")
    save_jsonl(valid_data, output_dir / "valid.jsonl")

    print("\n" + "=" * 50)
    print("Preprocessing completed!")
    print("=" * 50)

    # 打印统计信息
    print("\nData statistics:")
    print(f"  Total samples: {len(filtered_data)}")
    print(f"  Train samples: {len(train_data)} ({100 * len(train_data) / len(filtered_data):.1f}%)")
    print(f"  Valid samples: {len(valid_data)} ({100 * len(valid_data) / len(filtered_data):.1f}%)")


if __name__ == "__main__":
    main()
