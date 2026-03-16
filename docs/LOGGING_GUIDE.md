# 训练日志系统使用指南

## 概述

日志系统提供以下功能：
- **结构化日志** - JSON 格式的训练指标
- **TensorBoard 支持** - 可视化训练过程
- **自定义回调** - 实时格式化日志输出
- **训练报告** - 自动生成训练摘要
- **Benchmark 评测日志** - 统一的评测日志格式

## 输出文件结构

### 训练输出

```
outputs/
├── training_20260312_143022.log       # 文本日志
├── training_metrics.json              # 结构化指标 (JSON)
├── training_summary.json              # 训练摘要
├── train_info.json                    # 训练配置信息
├── training_report.txt                # 训练报告
├── trainer_state.json                 # 原始训练状态
├── checkpoint-100/                    # 模型检查点
└── tensorboard/                       # TensorBoard 日志
```

### Benchmark 评测输出

```
outputs/benchmark_eval/
├── benchmark_eval.log                 # 评测日志
├── benchmark_results.json             # 详细评测结果
├── benchmark_summary.json             # 评测摘要（多轮评测汇总）
├── results_mini.json                  # 单次评测详细结果
└── summary_mini.json                  # 单次评测摘要
```

## 日志文件说明

### 1. training_metrics.json

每条日志记录包含：
```json
{
  "step": 100,
  "epoch": 0.5,
  "loss": 2.3456,
  "learning_rate": 0.0002,
  "grad_norm": 1.23,
  "eval_loss": 2.1234,
  "eval_perplexity": 8.34,
  "gpu_memory_mb": 4521.5,
  "timestamp": "2026-03-12T14:30:22.123456"
}
```

### 2. training_summary.json

训练完成后生成：
```json
{
  "end_time": "2026-03-12T15:30:22",
  "total_time_seconds": 3600,
  "total_steps": 500,
  "steps_per_second": 0.14,
  "final_loss": 1.2345,
  "best_eval_loss": 1.1234
}
```

### 3. training_report.txt

人类可读的训练报告：
```
============================================================
SFT 训练报告
============================================================

【训练概览】
  训练步数：500
  总用时：3600.0 秒
  平均速度：0.14 steps/sec

【Loss 趋势】
  初始 Loss: 3.4567
  最终 Loss: 1.2345
  最小 Loss: 1.1234
  平均 Loss: 1.8765
  下降幅度：64.3%

【评估结果】
  评估次数：5
  最佳 eval_loss: 1.1234
  最终 eval_loss: 1.2345

【学习率变化】
  初始 LR: 2.00e-04
  最终 LR: 1.00e-04
```

## 使用方法

### 查看训练摘要

```bash
python scripts/view_logs.py --output_dir ./outputs --action summary
```

### 导出 CSV

```bash
python scripts/view_logs.py --output_dir ./outputs --action csv
```

### 绘制 Loss 曲线 (ASCII)

```bash
python scripts/view_logs.py --output_dir ./outputs --action plot
```

### 对比两次训练

```bash
python scripts/view_logs.py --output_dir ./outputs_v1 \
    --action compare --compare_with ./outputs_v2
```

### 查看最新日志

```bash
python scripts/view_logs.py --output_dir ./outputs --action tail --lines 50
```

## TensorBoard 使用

### 启动 TensorBoard

```bash
tensorboard --logdir ./outputs/tensorboard
```

然后在浏览器打开 http://localhost:6006

### 查看的指标

- `train/loss` - 训练损失
- `train/learning_rate` - 学习率变化
- `eval/eval_loss` - 评估损失
- `eval/eval_perplexity` - 困惑度

## 日志回调说明

### StructuredLoggingCallback

- 记录结构化指标到 JSON
- 实时打印格式化的训练日志
- 自动保存训练摘要

### TensorBoardCallback

- 将所有指标写入 TensorBoard
- 支持标量、标量对比图

## 控制台日志格式

训练时控制台输出：
```
14:30:22 | INFO     | ============================================================
14:30:22 | INFO     | 训练开始
14:30:22 | INFO     | ============================================================
14:30:22 | INFO     | 总步数：500
14:30:22 | INFO     | 训练轮次：3
14:30:22 | INFO     | 初始步数：0
14:31:45 | INFO     | [  100] | ep=0.20 | loss=2.3456 | lr=2.00e-04 | gpu=4521MB
14:33:02 | INFO     | [  200] | ep=0.40 | loss=1.8765 | lr=1.95e-04 | gpu=4521MB
```

## 配置选项

在 `configs/training_config.yaml` 中配置：

```yaml
training:
  logging_steps: 10        # 多少步打印一次日志
  save_steps: 100          # 多少步保存一次检查点
  output_dir: ./outputs    # 输出目录

# 新增日志配置
  use_tensorboard: true    # 是否启用 TensorBoard
  log_level: info          # 日志级别：debug, info, warning
```

## 编程方式使用

```python
from src.logger import setup_logger, generate_training_report

# 设置日志
logger = setup_logger("./outputs")
logger.info("训练开始")

# 训练完成后生成报告
report = generate_training_report("./outputs")
print(report)
```

## 日志分析示例

### 使用 Python 分析

```python
import json
import matplotlib.pyplot as plt

# 加载指标
with open("./outputs/training_metrics.json") as f:
    metrics = json.load(f)

# 绘制 Loss 曲线
steps = [m["step"] for m in metrics]
losses = [m["loss"] for m in metrics]

plt.figure(figsize=(10, 6))
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")
```

## 故障排查

### 日志文件为空

检查：
1. `logging_steps` 是否设置过小
2. 训练是否正常执行

### TensorBoard 无数据

检查：
1. 是否安装 `tensorboard`
2. `use_tensorboard` 是否为 `true`
3. 日志目录是否正确

### GPU 显存数据缺失

如果 `gpu_memory_mb` 始终为 null：
- 检查是否使用 CUDA
- 确认 PyTorch 是否正确安装


# ============================================================================
# Benchmark 评测日志系统
# ============================================================================

## 概述

Benchmark 评测日志系统提供：
- **统一的日志格式** - 所有评测脚本使用相同的日志格式
- **结构化输出** - JSON 格式的评测结果和摘要
- **进度跟踪** - 实时显示评测进度
- **时间记录** - 记录开始/结束时间和耗时
- **多轮对比** - 支持多次评测结果的汇总对比

## BenchmarkLogger API

### 基本使用

```python
from src.logger import BenchmarkLogger

# 初始化日志记录器
logger = BenchmarkLogger(output_dir="./outputs/benchmark_eval")

# 开始评测
logger.start_evaluation(benchmark_name="gsm8k", total_samples=200)

# 记录单个样本结果
logger.log_sample(
    idx=0,
    prompt="What is 1+1?",
    expected="2",
    predicted="2",
    correct=True,
    benchmark_name="gsm8k",
    qtype="math_word_problem"
)

# 记录进度
logger.log_progress(current=10, total=200, correct=8)

# 结束评测
metrics = logger.end_evaluation(
    total_samples=200,
    correct_samples=160
)
```

### 日志输出示例

控制台输出：
```
2026-03-12 15:30:22 | INFO     | ============================================================
2026-03-12 15:30:22 | INFO     | Benchmark 评测开始：gsm8k
2026-03-12 15:30:22 | INFO     | ============================================================
2026-03-12 15:30:22 | INFO     | 总样本数：200
2026-03-12 15:30:22 | INFO     | 开始时间：2026-03-12 15:30:22
2026-03-12 15:30:45 | INFO     | [1] 类型：math_word_problem
2026-03-12 15:30:45 | INFO     |   问题：Natalia sold clips to 48 of her friends...
2026-03-12 15:30:45 | INFO     |   期望：36
2026-03-12 15:30:45 | INFO     |   预测：36
2026-03-12 15:30:45 | INFO     |   结果：✓ 正确
2026-03-12 15:31:00 | INFO     | >>> 进度：[10/200] 正确率：80.0% <<<
```

### JSON 输出格式

**benchmark_results.json** - 详细评测结果：
```json
[
  {
    "idx": 0,
    "benchmark": "gsm8k",
    "type": "math_word_problem",
    "prompt": "Natalia sold clips to 48 of her friends...",
    "expected": "36",
    "predicted": "36",
    "correct": true,
    "timestamp": "2026-03-12T15:30:45.123456"
  }
]
```

**benchmark_summary.json** - 评测摘要汇总：
```json
{
  "gsm8k": {
    "total_samples": 200,
    "correct_samples": 160,
    "accuracy": 80.0,
    "start_time": "2026-03-12T15:30:22",
    "end_time": "2026-03-12T15:35:45",
    "duration_seconds": 323.5,
    "samples_per_second": 0.62,
    "category_breakdown": {}
  }
}
```

## 使用场景

### 1. eval_benchmark_local.py

本地模型评测：
```bash
python scripts/eval_benchmark_local.py \
    --base_model ./models/qwen2.5-0.5b \
    --adapter ./outputs/checkpoint-100 \
    --benchmark mini \
    --output_dir ./outputs/benchmark_eval
```

### 2. benchmark_eval.py

基于 lm-evaluation-harness 的评测：
```bash
python scripts/benchmark_eval.py \
    --base_model ./models/qwen2.5-0.5b \
    --adapter ./outputs/checkpoint-100 \
    --tasks gsm8k,arc,hellaswag \
    --output_dir ./outputs/benchmark_eval
```

### 3. download_benchmark_data.py

下载评测数据：
```bash
python scripts/download_benchmark_data.py
```

## 查看评测结果

### 查看评测摘要

```python
from src.logger import print_benchmark_summary

print_benchmark_summary("./outputs/benchmark_eval/benchmark_summary.json")
```

### 查看日志文件

```bash
# 查看评测日志
tail -f ./outputs/benchmark_eval/benchmark_eval.log

# 查看结构化结果
cat ./outputs/benchmark_eval/benchmark_results.json | jq '.[] | select(.correct == false)'
```

## 编程方式使用

```python
from src.logger import BenchmarkLogger, setup_logger

# 方式 1：使用 BenchmarkLogger
logger = BenchmarkLogger("./outputs/eval")
logger.start_evaluation("my_benchmark", total_samples=100)

# 在评测循环中记录
for i, item in enumerate(data):
    result = evaluate(item)
    logger.log_sample(
        idx=i,
        prompt=item["prompt"],
        expected=item["answer"],
        predicted=result,
        correct=result == item["answer"],
        benchmark_name="my_benchmark"
    )
    if (i + 1) % 10 == 0:
        logger.log_progress(i + 1, len(data), sum(...))

# 结束评测
logger.end_evaluation(total=len(data), correct=sum(...))

# 方式 2：使用基础 logger
logger = setup_logger("./outputs", name="custom_eval")
logger.info("开始评测...")
logger.info(f"样本数：{len(data)}")
```

## Benchmark 日志指标说明

| 指标 | 说明 |
|------|------|
| `total_samples` | 总评测样本数 |
| `correct_samples` | 正确预测的样本数 |
| `accuracy` | 正确率（百分比） |
| `start_time` | 评测开始时间 |
| `end_time` | 评测结束时间 |
| `duration_seconds` | 总耗时（秒） |
| `samples_per_second` | 评测速度（样本/秒） |
| `category_breakdown` | 按类别的分类统计 |
