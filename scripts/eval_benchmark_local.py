#!/usr/bin/env python3
"""
使用下载的 benchmark 数据进行本地评测
"""
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# 强制离线模式，避免huggingface_hub将本地路径误认为远程repo id
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import yaml

from src.model import load_tokenizer, load_base_model
from src.evaluator import SFTEvaluator
from src.logger import BenchmarkLogger, print_benchmark_summary
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on benchmark data")

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
        "--benchmark",
        type=str,
        default="mini",
        choices=["mini", "chinese_basic", "gsm8k", "arc_easy", "arc_challenge", "hellaswag", "truthfulqa", "mmlu", "all"],
        help="评测 benchmark（mini=快速评测，all=全量评测）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大评测样本数（默认评测全部）",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="起始评测样本索引（默认从0开始）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果保存目录（默认: output/log/benchmark/日期_模型）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="强制使用指定设备（默认: cuda如果可用，否则cpu）",
    )

    return parser.parse_args()


def get_model_name_from_adapter(adapter_path: str) -> str:
    """从adapter路径中提取模型名称"""
    # 例如: ./outputs/lora/alpaca1000 -> alpaca1000
    # 例如: ./outputs/alpaca_1000 -> alpaca_1000
    path = Path(adapter_path)
    return path.name


def extract_answer(text: str, benchmark: str) -> str:
    """从生成的文本中提取答案"""
    text = text.strip()

    if benchmark == "gsm8k":
        # 提取最后一个数字
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return numbers[-1] if numbers else text

    elif benchmark in ["arc_easy", "arc_challenge", "hellaswag", "truthfulqa", "mmlu", "chinese_basic", "mini_benchmarks"]:
        # 提取 A/B/C/D 答案

        # 0. 先处理特殊情况：如果文本以 "A\n" 或 "A." 开头，直接返回
        first_line = text.split('\n')[0].strip()
        match = re.match(r'^([A-D])\.?\s*$', first_line, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 1. 首先查找 "The correct answer is X" 或 "Answer is X" 格式
        match = re.search(r'[Tt]he correct answer is\s*([A-D])[.\s]?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r'[Aa]nswer is\s*([A-D])[.\s]?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 2. 查找 "Option X" 或 "选项 X" 格式
        match = re.search(r'[Oo]ption?\s*([A-D])[.\s]?', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 3. 查找任意位置的 "A." 或 "A " 或 "(A)" 格式（不限行首）
        match = re.search(r'([A-D])[.\s\)]', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # 4. 找第一个单独的大写字母 A/B/C/D（通常是答案）
        letters = re.findall(r'\b([A-D])\b', text.upper())
        if letters:
            return letters[0]  # 返回第一个，通常是答案

    return text


def evaluate_chinese_basic(evaluator, data, benchmark_logger: BenchmarkLogger, max_samples=50):
    """评估中文基础题（混合题型）"""
    results = []
    correct = 0

    for i, item in enumerate(data[:max_samples]):
        prompt = item["prompt"]
        expected = item["answer"]
        qtype = item.get("type", "instruction_following")

        # 不限制 max_new_tokens，让模型完整输出
        response = evaluator.generate(prompt, temperature=0.1, max_new_tokens=1024)

        # 根据题型判断正确性
        if qtype == "multiple_choice":
            predicted = extract_answer(response, "chinese_basic")
            is_correct = predicted == expected.upper()
        elif qtype == "math_word_problem":
            # 提取数字答案
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            predicted = numbers[-1] if numbers else response
            is_correct = predicted == expected
        else:
            # 开放式问题，简单匹配或使用 LLM 评判
            # 这里简化处理：检查是否生成了合理长度的回答
            is_correct = len(response.strip()) > 5
            predicted = response  # 保存完整响应，不做截断

        if is_correct:
            correct += 1

        # 使用日志记录
        benchmark_logger.log_sample(
            idx=i,
            prompt=prompt,
            expected=expected,
            predicted=predicted,
            correct=is_correct,
            benchmark_name=benchmark_logger.current_benchmark,
            qtype=qtype,
            response=response
        )

        results.append({
            "idx": i,
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "response": response,  # 保存完整响应
            "correct": is_correct,
            "type": qtype,
        })

        if (i + 1) % 10 == 0:
            benchmark_logger.log_progress(i + 1, max_samples if max_samples else len(data), correct)

    return results, correct


def evaluate_gsm8k(evaluator, data, benchmark_logger: BenchmarkLogger, max_samples=50):
    """评估 GSM8K 数学题"""
    results = []
    correct = 0

    for i, item in enumerate(data[:max_samples]):
        prompt = item["prompt"]
        expected = item["answer"]

        # 不限制 max_new_tokens，让模型完整输出
        response = evaluator.generate(prompt, temperature=0.1, max_new_tokens=1024)
        predicted = extract_answer(response, "gsm8k")

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        # 使用日志记录
        benchmark_logger.log_sample(
            idx=i,
            prompt=prompt,
            expected=expected,
            predicted=predicted,
            correct=is_correct,
            benchmark_name=benchmark_logger.current_benchmark,
            qtype="math_word_problem",
            response=response
        )

        results.append({
            "idx": i,
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "response": response,  # 保存完整响应
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            benchmark_logger.log_progress(i + 1, max_samples if max_samples else len(data), correct)

    return results, correct


def evaluate_multiple_choice(evaluator, data, benchmark_logger: BenchmarkLogger, max_samples=50, start_index=0):
    """评估选择题（ARC/Hellaswag/TruthfulQA/MMLU）"""
    results = []
    correct = 0

    # 根据起始索引切片数据
    eval_data = data[start_index:start_index + max_samples] if max_samples else data[start_index:]
    benchmark_logger.logger.info(f"从第 {start_index + 1} 题开始评测，共 {len(eval_data)} 题")

    for i, item in enumerate(eval_data):
        prompt = item["prompt"]
        expected = item["answer"].upper()

        # 不限制 max_new_tokens，让模型完整输出
        response = evaluator.generate(prompt, temperature=0.1, max_new_tokens=1024)
        predicted = extract_answer(response, benchmark_logger.current_benchmark)

        is_correct = predicted == expected
        if is_correct:
            correct += 1

        # 使用日志记录
        benchmark_logger.log_sample(
            idx=i,
            prompt=prompt,
            expected=expected,
            predicted=predicted,
            correct=is_correct,
            benchmark_name=benchmark_logger.current_benchmark,
            qtype="multiple_choice",
            response=response
        )

        results.append({
            "idx": i,
            "prompt": prompt,  # 完整 prompt，不做截断
            "expected": expected,
            "predicted": predicted,
            "response": response,  # 保存完整响应
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            benchmark_logger.log_progress(i + 1, max_samples if max_samples else len(data), correct)

    return results, correct


def main():
    args = parse_args()

    # 获取模型名称
    model_name = get_model_name_from_adapter(args.adapter)

    # 生成日期时间戳
    timestamp = datetime.now().strftime("%Y%m%d")

    # 设置输出目录: output/log/benchmark/日期_模型
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"./outputs/log/benchmark/{timestamp}_{model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = BenchmarkLogger(output_dir, name="eval_benchmark_local")

    print("\n" + "=" * 60)
    print("Benchmark Evaluation (Local)")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Model: {model_name}")
    if args.benchmark == "mini":
        print(f"Mode: QUICK (mini benchmarks ~180 samples)")
    elif args.benchmark == "all":
        print(f"Mode: FULL (all benchmarks ~1600 samples)")
    else:
        print(f"Benchmark: {args.benchmark}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print(f"Output dir: {output_dir}")
    print(f"Log file: {output_dir}/benchmark_eval.log")
    print("=" * 60)

    # 加载模型
    logger.logger.info("开始加载模型...")
    tokenizer = load_tokenizer(args.base_model)
    base_model = load_base_model(args.base_model, torch_dtype=torch.bfloat16)

    # 如果 adapter 存在则加载，否则直接使用基座模型
    adapter_path = Path(args.adapter).resolve()
    if args.adapter and adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        logger.logger.info(f"已加载 LoRA adapter: {args.adapter}")
    else:
        model = base_model
        logger.logger.info("使用基座模型（无 adapter）")

    # 确定设备
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.logger.info(f"使用设备: {device}")
    model.to(device)
    model.eval()

    evaluator = SFTEvaluator(model, tokenizer, max_length=1024, device=device)

    # 确定要评测的 benchmark
    if args.benchmark == "all":
        # 全量评测
        benchmarks = ["chinese_basic", "gsm8k", "arc_easy", "hellaswag"]
    elif args.benchmark == "mini":
        # 快速评测（精简版）
        benchmarks = ["mini_benchmarks"]
    else:
        benchmarks = [args.benchmark]

    all_results = {}
    summary = {}

    for benchmark in benchmarks:
        logger.logger.info(f"\n{'='*60}")
        logger.logger.info(f"开始评测：{benchmark}")
        logger.logger.info(f"{'='*60}")

        # 开始评测记录
        logger.start_evaluation(benchmark, 0)

        # 加载数据
        data_path = Path(f"./bench_test_data/{benchmark}.jsonl")
        if not data_path.exists():
            # 尝试 mini 版本
            mini_path = Path(f"./bench_test_data/{benchmark}_mini.jsonl")
            if mini_path.exists():
                data_path = mini_path
            else:
                logger.logger.warning(f"未找到数据文件：{data_path}，跳过...")
                continue

        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        logger.logger.info(f"加载了 {len(data)} 个样本")
        num_samples = min(args.max_samples, len(data)) if args.max_samples else len(data)
        logger.logger.info(f"评测 {num_samples} 个样本...")

        # 更新评测样本数
        logger.start_time = time.time()
        logger.current_benchmark = benchmark

        # 评估
        if benchmark == "gsm8k":
            results, correct = evaluate_gsm8k(evaluator, data, logger, args.max_samples)
        elif benchmark == "chinese_basic":
            results, correct = evaluate_chinese_basic(evaluator, data, logger, args.max_samples)
        else:
            results, correct = evaluate_multiple_choice(evaluator, data, logger, args.max_samples, args.start_index)

        total = len(results)
        accuracy = correct / total * 100 if total > 0 else 0

        # 结束评测记录
        logger.end_evaluation(total, correct)

        logger.logger.info(f"\n  结果：{correct}/{total} = {accuracy:.2f}%")

        all_results[benchmark] = results
        summary[benchmark] = {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }

    # 保存详细结果
    results_path = output_dir / f"results_{args.benchmark}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印总结
    print("\n" + "=" * 60)
    print("评测总结")
    print("=" * 60)
    for benchmark, stats in summary.items():
        print(f"  {benchmark}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print("=" * 60)
    print(f"\n详细结果保存至：{results_path}")
    print(f"日志文件：{output_dir}/benchmark_eval.log")

    # 打印评测摘要
    print_benchmark_summary(output_dir / "benchmark_summary.json")


if __name__ == "__main__":
    main()
