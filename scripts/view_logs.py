#!/usr/bin/env python3
"""
查看训练日志工具
支持查看结构化日志、生成图表、对比多次训练
支持查看 Benchmark 评测日志
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.logger import print_benchmark_summary


def load_metrics(output_dir: str) -> list:
    """加载训练指标"""
    metrics_file = Path(output_dir) / "training_metrics.json"
    if not metrics_file.exists():
        print(f"错误：未找到训练指标文件 {metrics_file}")
        return []

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def show_summary(output_dir: str):
    """显示训练摘要"""
    metrics = load_metrics(output_dir)
    if not metrics:
        return

    print("\n" + "=" * 60)
    print("训练摘要")
    print("=" * 60)

    # 基本信息
    print(f"\n总步数：{len(metrics)}")
    print(f"起始时间：{metrics[0]['timestamp'][:19] if metrics else 'N/A'}")

    # Loss 统计
    losses = [m.get('loss', 0) for m in metrics]
    print(f"\nLoss 统计:")
    print(f"  初始 Loss: {losses[0]:.4f}")
    print(f"  最终 Loss: {losses[-1]:.4f}")
    print(f"  最小 Loss: {min(losses):.4f}")
    print(f"  平均 Loss: {sum(losses)/len(losses):.4f}")
    print(f"  下降幅度：{(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

    # 评估统计
    eval_losses = [m.get('eval_loss') for m in metrics if m.get('eval_loss')]
    if eval_losses:
        print(f"\n评估统计:")
        print(f"  评估次数：{len(eval_losses)}")
        print(f"  最佳 eval_loss: {min(eval_losses):.4f}")
        print(f"  最终 eval_loss: {eval_losses[-1]:.4f}")

    # GPU 显存统计
    gpu_memories = [m.get('gpu_memory_mb') for m in metrics if m.get('gpu_memory_mb')]
    if gpu_memories:
        print(f"\nGPU 显存:")
        print(f"  平均：{sum(gpu_memories)/len(gpu_memories):.0f} MB")
        print(f"  峰值：{max(gpu_memories):.0f} MB")

    print("\n" + "=" * 60)


def export_csv(output_dir: str):
    """导出为 CSV 格式"""
    metrics = load_metrics(output_dir)
    if not metrics:
        return

    output_file = Path(output_dir) / "training_metrics.csv"

    # 确定所有字段
    all_keys = set()
    for m in metrics:
        all_keys.update(m.keys())

    with open(output_file, "w", encoding="utf-8") as f:
        # 写入表头
        headers = sorted(all_keys)
        f.write(",".join(headers) + "\n")

        # 写入数据
        for m in metrics:
            values = [str(m.get(h, "")) for h in headers]
            f.write(",".join(values) + "\n")

    print(f"已导出 CSV 到：{output_file}")


def plot_loss(output_dir: str):
    """绘制 Loss 曲线图 (ASCII)"""
    metrics = load_metrics(output_dir)
    if not metrics:
        return

    losses = [m.get('loss', 0) for m in metrics]

    # 简单 ASCII 图表
    print("\nLoss 曲线 (ASCII)")
    print("-" * 60)

    # 采样显示 (最多 50 个点)
    step = max(1, len(losses) // 50)
    sampled = losses[::step][:50]

    min_loss = min(sampled)
    max_loss = max(sampled)
    range_loss = max_loss - min_loss or 1

    height = 10
    for row in range(height, -1, -1):
        threshold = min_loss + (row / height) * range_loss
        line = ""
        for val in sampled:
            if val >= threshold:
                line += "█"
            else:
                line += " "
        print(f"{threshold:6.3f} | {line}")

    print(" " * 7 + "+" + "-" * len(sampled))
    print(" " * 7 + f"0 → {len(losses)} steps")


def compare_trainings(dir1: str, dir2: str):
    """对比两次训练"""
    metrics1 = load_metrics(dir1)
    metrics2 = load_metrics(dir2)

    if not metrics1 or not metrics2:
        return

    print("\n" + "=" * 60)
    print("训练对比")
    print("=" * 60)

    losses1 = [m.get('loss', 0) for m in metrics1]
    losses2 = [m.get('loss', 0) for m in metrics2]

    print(f"\n{'指标':<20} {'训练 1':<15} {'训练 2':<15}")
    print("-" * 50)
    print(f"{'总步数':<20} {len(metrics1):<15} {len(metrics2):<15}")
    print(f"{'初始 Loss':<20} {losses1[0]:.4f}{'':<10} {losses2[0]:.4f}{'':<10}")
    print(f"{'最终 Loss':<20} {losses1[-1]:.4f}{'':<10} {losses2[-1]:.4f}{'':<10}")
    print(f"{'最小 Loss':<20} {min(losses1):.4f}{'':<10} {min(losses2):.4f}{'':<10}")
    print(f"{'平均 Loss':<20} {sum(losses1)/len(losses1):.4f}{'':<10} {sum(losses2)/len(losses2):.4f}{'':<10}")

    # 下降幅度对比
    drop1 = (losses1[0] - losses1[-1]) / losses1[0] * 100
    drop2 = (losses2[0] - losses2[-1]) / losses2[0] * 100
    print(f"{'下降幅度':<20} {drop1:.1f}%{'':<14} {drop2:.1f}%{'':<14}")

    print("\n" + "=" * 60)


def tail_log(output_dir: str, lines: int = 20):
    """显示最近的日志"""
    # 查找最新的 log 文件
    log_dir = Path(output_dir)
    log_files = sorted(log_dir.glob("training_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not log_files:
        print("未找到训练日志文件")
        return

    latest_log = log_files[0]
    print(f"\n查看最新日志：{latest_log.name}")
    print("-" * 60)

    with open(latest_log, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        for line in all_lines[-lines:]:
            print(line, end="")


def show_benchmark_summary(output_dir: str):
    """显示 Benchmark 评测摘要"""
    summary_file = Path(output_dir) / "benchmark_summary.json"

    if not summary_file.exists():
        # 尝试父目录
        summary_file = Path(output_dir).parent / "benchmark_summary.json"

    if not summary_file.exists():
        print("错误：未找到 Benchmark 摘要文件")
        return

    print_benchmark_summary(summary_file)


def show_benchmark_results(output_dir: str, filter_correct: bool = None, limit: int = 10):
    """显示 Benchmark 评测详细结果"""
    results_file = Path(output_dir) / "benchmark_results.json"

    if not results_file.exists():
        print("错误：未找到 Benchmark 结果文件")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 筛选
    if filter_correct is not None:
        results = [r for r in results if r.get("correct") == filter_correct]

    # 限制显示数量
    results = results[:limit]

    print("\n" + "=" * 60)
    print(f"Benchmark 详细结果 (显示 {len(results)} 条)")
    print("=" * 60)

    for r in results:
        benchmark = r.get("benchmark", "unknown")
        idx = r.get("idx", 0)
        correct = r.get("correct", False)
        expected = r.get("expected", "")
        predicted = r.get("predicted", "")
        prompt = r.get("prompt", "")[:80]

        status = "✓" if correct else "✗"
        print(f"\n[{status}] {benchmark} #{idx}")
        print(f"  问题：{prompt}...")
        print(f"  期望：{expected}")
        print(f"  预测：{predicted}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="查看 SFT 训练日志和 Benchmark 评测日志")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="训练输出目录")
    parser.add_argument("--action", type=str,
                       choices=["summary", "csv", "plot", "compare", "tail",
                               "bench_summary", "bench_results"],
                       default="summary", help="操作类型")
    parser.add_argument("--compare_with", type=str, help="对比的训练目录 (用于 compare 模式)")
    parser.add_argument("--lines", type=int, default=20, help="显示日志行数 (用于 tail 模式)")
    parser.add_argument("--filter_correct", type=str, choices=["true", "false"],
                       help="筛选正确/错误的结果 (用于 bench_results 模式)")
    parser.add_argument("--limit", type=int, default=10, help="显示结果数量限制 (用于 bench_results 模式)")

    args = parser.parse_args()

    if args.action == "summary":
        show_summary(args.output_dir)
    elif args.action == "csv":
        export_csv(args.output_dir)
    elif args.action == "plot":
        plot_loss(args.output_dir)
    elif args.action == "compare":
        if args.compare_with:
            compare_trainings(args.output_dir, args.compare_with)
        else:
            print("错误：compare 模式需要指定 --compare_with 参数")
    elif args.action == "tail":
        tail_log(args.output_dir, args.lines)
    elif args.action == "bench_summary":
        show_benchmark_summary(args.output_dir)
    elif args.action == "bench_results":
        filter_correct = None
        if args.filter_correct == "true":
            filter_correct = True
        elif args.filter_correct == "false":
            filter_correct = False
        show_benchmark_results(args.output_dir, filter_correct=filter_correct, limit=args.limit)


if __name__ == "__main__":
    main()
