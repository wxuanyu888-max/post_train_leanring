#!/usr/bin/env python3
"""生成模型对比图表"""

import matplotlib.pyplot as plt
import numpy as np

# 模型数据
models = ['Baseline', 'SFT-Alpaca\n(Alpaca1000)', 'SFT-SFT1000', 'SFT-V1', 'DPO-V1']
accuracies = [17.22, 17.78, 21.11, 17.78, 20.56]
correct = [31, 32, 38, 32, 37]
total = 180

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 图1: 准确率对比
colors = ['#808080', '#4CAF50', '#2196F3', '#FF9800', '#E91E63']
bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)

# 添加数值标签
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.annotate(f'{acc}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 25)
ax1.axhline(y=17.22, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# 图2: 正确数对比
bars2 = ax2.bar(models, correct, color=colors, edgecolor='black', linewidth=1.2)

# 添加数值标签
for bar, c in zip(bars2, correct):
    height = bar.get_height()
    ax2.annotate(f'{c}/{total}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Correct Answers', fontsize=12)
ax2.set_title('Correct Answers (out of 180)', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 45)
ax2.axhline(y=31, color='gray', linestyle='--', alpha=0.5, label='Baseline')

plt.tight_layout()
plt.savefig('./docs/training_results.png', dpi=150, bbox_inches='tight')
print("图表已保存到 ./docs/training_results.png")
