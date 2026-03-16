#!/usr/bin/env python3
"""
评估脚本
评估微调后模型的性能
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from src.model import load_tokenizer, load_base_model
from src.evaluator import SFTEvaluator
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT model")

    parser.add_argument(
        "--base_model",
        type=str,
        default="./models/qwen2.5-0.5b-instruct",
        help="基座模型路径",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./outputs",
        help="LoRA Adapter 路径",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="./data/processed/valid.jsonl",
        help="评估数据路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/eval_results.json",
        help="评估结果保存路径",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评估样本数",
    )

    return parser.parse_args()


def load_eval_data(path: str, max_samples: int = None) -> list:
    """加载评估数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if max_samples:
        data = data[:max_samples]

    return data


def main():
    args = parse_args()

    print("=" * 60)
    print("SFT Model Evaluation")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Eval data: {args.eval_data}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # 加载分词器
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(args.base_model)

    # 加载基座模型
    print("Loading base model...")
    base_model = load_base_model(
        args.base_model,
        torch_dtype=torch.bfloat16,
    )

    # 加载 Adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 加载评估数据
    print("\nLoading evaluation data...")
    eval_data = load_eval_data(args.eval_data, args.max_samples)
    print(f"Evaluation samples: {len(eval_data)}")

    # 创建评估器
    evaluator = SFTEvaluator(
        model,
        tokenizer,
        max_length=512,
    )

    # 开始评估
    print("\nStarting evaluation...")
    results = evaluator.evaluate_dataset(
        eval_data,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # 计算指标
    metrics = evaluator.compute_metrics(results["results"])

    # 打印指标
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Avg generated length: {metrics['avg_generated_length']:.1f}")
    print(f"Avg reference length: {metrics['avg_reference_length']:.1f}")
    print("=" * 60)

    # 保存指标
    metrics_path = Path(args.output).parent / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")
    print(f"Metrics saved to: {metrics_path}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
