#!/usr/bin/env python3
"""
通用能力评测脚本
支持主流 Benchmark: C-Eval, CMMLU, MMLU, GSM8K, ARC, Hellaswag 等
基于 lm-evaluation-harness 框架
"""
import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

from src.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SFT model on benchmarks")

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
        "--tasks",
        type=str,
        default="ceval,cmmlu,gsm8k,arc",
        help="评测任务，逗号分隔 (ceval,cmmlu,mmlu,gsm8k,arc,hellaswag,truthfulqa)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/benchmark_results",
        help="结果保存目录",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="批处理大小",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=5,
        help="Few-shot 样本数",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制每个任务的最大样本数 (用于快速测试)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="信任远程代码",
    )

    return parser.parse_args()


def check_lm_eval_installed():
    """检查 lm-eval 是否安装"""
    try:
        import lm_eval
        print(f"lm-eval version: {lm_eval.__version__}")
        return True
    except ImportError:
        print("\n" + "=" * 60)
        print("错误：lm-eval 未安装")
        print("=" * 60)
        print("\n请运行以下命令安装:")
        print("  pip install lm-eval[torch]")
        print("\n或者使用 HuggingFace 版本:")
        print("  pip install lm-eval")
        print("=" * 60 + "\n")
        return False


def run_benchmark(args):
    """运行 lm-evaluation-harness"""
    # 初始化日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, name="benchmark_eval")

    start_time = time.time()

    # 构建 model_args
    model_args = f"pretrained={args.base_model}"
    if args.adapter and Path(args.adapter).exists():
        model_args += f",peft={args.adapter}"
    if args.trust_remote_code:
        model_args += ",trust_remote_code=True"

    # 构建任务列表
    tasks = args.tasks.replace(" ", "")

    # 构建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 构建命令
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--batch_size", str(args.batch_size),
        "--num_fewshot", str(args.num_fewshot),
        "--output_path", args.output_dir,
        "--write_out",
    ]

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    logger.info("=" * 60)
    logger.info("Benchmark Evaluation")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Adapter: {args.adapter}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Few-shot: {args.num_fewshot}")
    logger.info(f"Output dir: {args.output_dir}")
    if args.limit:
        logger.info(f"Limit: {args.limit} samples per task (quick test)")
    logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    logger.info(f"\nRunning command:")
    logger.info(" ".join(cmd))

    # 运行评估
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation completed!")
        logger.info("=" * 60)
        logger.info(f"Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        # 查找并显示结果
        results_file = Path(args.output_dir) / "results.json"
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            logger.info("\n" + "=" * 60)
            logger.info("Results Summary")
            logger.info("=" * 60)

            for task_name, task_results in results.get("results", {}).items():
                logger.info(f"\n{task_name}:")
                for metric, value in task_results.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
                    else:
                        logger.info(f"  {metric}: {value}")

            # 保存日志摘要
            summary = {
                "base_model": args.base_model,
                "adapter": args.adapter,
                "tasks": tasks,
                "batch_size": args.batch_size,
                "num_fewshot": args.num_fewshot,
                "limit": args.limit,
                "duration_seconds": elapsed,
                "end_time": time.strftime('%Y-%m-%d %H:%M:%S'),
                "results": results.get("results", {}),
            }
            summary_file = output_dir / "benchmark_eval_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            logger.info(f"\n评测摘要已保存：{summary_file}")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"\nEvaluation failed after {elapsed:.1f} seconds: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"\nError after {elapsed:.1f} seconds: {e}")
        return False


def show_available_tasks():
    """显示可用的评测任务"""
    print("\n" + "=" * 60)
    print("Available Benchmark Tasks")
    print("=" * 60)

    tasks = {
        "中文能力": [
            ("ceval", "C-Eval 中学到专业考试，52 个学科"),
            ("cmmlu", "CMMLU 中文多选，67 个主题"),
            ("agieval", "AGIEval 高考/司法考试"),
        ],
        "综合知识": [
            ("mmlu", "MMLU 57 学科综合知识，14K 题"),
            ("truthfulqa", "TruthfulQA 事实性/抗幻觉"),
        ],
        "推理能力": [
            ("gsm8k", "GSM8K 小学数学应用题"),
            ("math", "MATH 高中到竞赛级数学"),
            ("arc_easy", "ARC 简单科学问答"),
            ("arc_challenge", "ARC 挑战级科学问答"),
        ],
        "语言理解": [
            ("hellaswag", "Hellaswag 常识推理"),
            ("winogrande", "Winogrande 指代消解"),
            ("copa", "COPA 因果推理"),
        ],
        "代码能力": [
            ("humaneval", "HumanEval 164 道编程题"),
            ("mbpp", "MBPP 974 道 Python 编程题"),
        ],
    }

    for category, task_list in tasks.items():
        print(f"\n{category}:")
        for task_name, description in task_list:
            print(f"  --tasks {task_name:20s} {description}")

    print("\n" + "=" * 60)
    print("\n示例:")
    print("  # 中文能力评测")
    print("  python benchmark_eval.py --tasks ceval,cmmlu")
    print("\n  # 综合能力评测")
    print("  python benchmark_eval.py --tasks mmlu,gsm8k,arc,hellaswag")
    print("\n  # 快速测试 (每个任务 100 题)")
    print("  python benchmark_eval.py --tasks mmlu --limit 100")
    print("=" * 60 + "\n")


def main():
    # 检查是否显示了可用任务
    if "--list-tasks" in sys.argv or "-h" in sys.argv:
        show_available_tasks()
        return

    args = parse_args()

    # 检查 lm-eval 安装
    if not check_lm_eval_installed():
        print("\n提示：使用 --list-tasks 查看可用的评测任务")
        sys.exit(1)

    # 运行评测
    success = run_benchmark(args)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
