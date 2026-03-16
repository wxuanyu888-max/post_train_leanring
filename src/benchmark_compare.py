#!/usr/bin/env python3
"""
Benchmark 对比分析脚本
对比不同模型在 benchmark 上的表现，并绘制可视化图表
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_results(result_path: str):
    """加载评测结果，返回 {idx: correct} 的字典"""
    if not os.path.exists(result_path):
        print(f"警告: 找不到文件 {result_path}")
        return {}

    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 处理不同格式
    results = {}
    if isinstance(data, dict):
        # 新格式: {"mini_benchmarks": [...]}
        for key in data.keys():
            for item in data[key]:
                idx = item['idx']
                results[idx] = item.get('correct', False)
    elif isinstance(data, list):
        # 旧格式: [...]
        for item in data:
            idx = item['idx']
            results[idx] = item.get('correct', False)

    return results


def load_all_benchmarks():
    """加载所有 benchmark 结果"""
    base_dir = Path("./outputs/log/benchmark")

    benchmarks = {
        "baseline": {
            "path": base_dir / "20260312_baseline" / "results.json",
            "label": "Baseline (基线)",
            "color": "#808080",  # 灰色
        },
        "sft1000": {
            "path": base_dir / "20260313_sft1000" / "results.json",
            "label": "SFT-1000",
            "color": "#3498db",  # 蓝色
        },
        "dpo_v1": {
            "path": base_dir / "20260313_dpo_v1" / "results_mini.json",
            "label": "DPO-V1",
            "color": "#e74c3c",  # 红色
        },
    }

    # 加载每个 benchmark
    results = {}
    for name, config in benchmarks.items():
        correct_set = set()
        results[name] = {
            'config': config,
            'correct_indices': set(),
            'wrong_indices': set(),
            'total': 0,
            'correct': 0,
        }

        data = load_results(str(config['path']))

        for idx, is_correct in data.items():
            results[name]['total'] += 1
            if is_correct:
                results[name]['correct_indices'].add(idx)
                results[name]['correct'] += 1
            else:
                results[name]['wrong_indices'].add(idx)

    return results


def analyze_overlap(results):
    """分析各模型之间的重叠情况"""
    # 找出所有模型都做对的题目
    all_correct = None
    for name, data in results.items():
        if all_correct is None:
            all_correct = data['correct_indices'].copy()
        else:
            all_correct &= data['correct_indices']

    # 找出每个模型独有的正确题目
    unique_correct = {}
    for name, data in results.items():
        others_correct = set()
        for other_name, other_data in results.items():
            if other_name != name:
                others_correct |= other_data['correct_indices']
        unique_correct[name] = data['correct_indices'] - others_correct

    return all_correct, unique_correct


def print_comparison_report(results):
    """打印对比报告"""
    print("=" * 70)
    print("Benchmark 对比分析报告")
    print("=" * 70)

    # 打印各模型总体准确率
    print("\n## 总体准确率\n")
    print(f"| 模型 | 正确数 | 总数 | 准确率 |")
    print(f"|------|--------|------|--------|")
    for name, data in results.items():
        acc = data['correct'] / data['total'] * 100 if data['total'] > 0 else 0
        print(f"| {data['config']['label']:<20} | {data['correct']:>6} | {data['total']:>4} | {acc:>6.2f}% |")

    # 分析重叠
    all_correct, unique_correct = analyze_overlap(results)

    print(f"\n## 各模型重叠分析\n")
    print(f"所有模型都做对的题目数: {len(all_correct)}")
    print(f"题目索引: {sorted(all_correct)}")

    print("\n各模型独有的正确题目:")
    for name, indices in unique_correct.items():
        label = results[name]['config']['label']
        print(f"  {label}: {len(indices)} 题 -> {sorted(indices)}")

    # 打印每道题在各模型中的对错情况
    print(f"\n## 详细对比表\n")
    print(f"| 题号 | " + " | ".join([results[name]['config']['label'][:10] for name in results]) + " |")
    print(f"|------|" + "|".join(["------" for _ in results]) + "|")

    # 获取所有题目的索引
    all_indices = set()
    for data in results.values():
        all_indices |= data['correct_indices']
        all_indices |= data['wrong_indices']

    for idx in sorted(all_indices):
        row = f"| {idx:>3}  |"
        for name in results.keys():
            correct = "✓" if idx in results[name]['correct_indices'] else "✗"
            row += f" {correct:^5} |"
        print(row)


def plot_visualization(results, output_path: str = "./outputs/log/benchmark/comparison.png"):
    """绘制可视化图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 准确率对比柱状图
    ax1 = axes[0, 0]
    names = [results[name]['config']['label'] for name in results]
    accuracies = [results[name]['correct'] / results[name]['total'] * 100 for name in results]
    colors = [results[name]['config']['color'] for name in results]

    bars = ax1.bar(names, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 30)

    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)

    # 2. 韦恩图风格的题目重叠分析
    ax2 = axes[0, 1]
    ax2.set_title('Correct Questions Overlap', fontsize=14, fontweight='bold')

    # 绘制每个模型的正确题目数量
    x = range(len(results))
    correct_counts = [results[name]['correct'] for name in results]
    bars2 = ax2.bar(x, correct_counts, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('Number of Correct Questions', fontsize=12)

    for bar, count in zip(bars2, correct_counts):
        ax2.annotate(str(count),
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. 题目对错热力图
    ax3 = axes[1, 0]

    # 获取所有题目的索引范围
    all_indices = set()
    for data in results.values():
        all_indices |= data['correct_indices']
        all_indices |= data['wrong_indices']
    max_idx = max(all_indices) + 1

    # 创建热力图数据
    matrix = []
    for name in results:
        row = []
        for idx in range(max_idx):
            if idx in results[name]['correct_indices']:
                row.append(1)
            elif idx in results[name]['wrong_indices']:
                row.append(0)
            else:
                row.append(-1)  # 未评测
        matrix.append(row)

    # 绘制热力图
    im = ax3.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names)
    ax3.set_xlabel('Question Index', fontsize=12)
    ax3.set_title('Question-by-Question Results (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')

    # 4. 各区间正确率对比
    ax4 = axes[1, 1]

    # 按区间统计
    interval_size = 20
    num_intervals = (max_idx + interval_size - 1) // interval_size

    for name in results:
        interval_acc = []
        interval_labels = []
        for i in range(num_intervals):
            start = i * interval_size
            end = min((i + 1) * interval_size, max_idx)
            interval_labels.append(f'{start}-{end-1}')

            correct = sum(1 for idx in range(start, end)
                         if idx in results[name]['correct_indices'])
            total = sum(1 for idx in range(start, end)
                       if idx in results[name]['correct_indices'] or idx in results[name]['wrong_indices'])
            acc = correct / total * 100 if total > 0 else 0
            interval_acc.append(acc)

        ax4.plot(interval_labels, interval_acc, 'o-',
                label=results[name]['config']['label'],
                color=results[name]['config']['color'],
                linewidth=2, markersize=6)

    ax4.set_xlabel('Question Interval', fontsize=12)
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Accuracy by Question Interval', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.set_ylim(0, 60)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")


def export_detail_csv(results, output_path: str = "./outputs/log/benchmark/comparison_details.csv"):
    """导出详细对比 CSV"""
    import csv

    # 获取所有题目的索引
    all_indices = set()
    for data in results.values():
        all_indices |= data['correct_indices']
        all_indices |= data['wrong_indices']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        header = ['Question_Index']
        for name in results:
            header.append(f'{name}_Correct')
        writer.writerow(header)

        # 写入数据
        for idx in sorted(all_indices):
            row = [idx]
            for name in results:
                row.append(1 if idx in results[name]['correct_indices'] else 0)
            writer.writerow(row)

    print(f"详细对比数据已保存至: {output_path}")


def main():
    print("开始加载评测结果...")
    results = load_all_benchmarks()

    print_comparison_report(results)

    # 绘制可视化
    plot_visualization(results)

    # 导出 CSV
    export_detail_csv(results)

    print("\n对比分析完成!")


if __name__ == "__main__":
    main()
