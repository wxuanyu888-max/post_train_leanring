# Benchmark 评测数据集

本目录包含从主流 LLM 评测基准中采样的测试数据，用于快速评估模型性能。

## 数据集概览

### 精简版（推荐）⭐

| Benchmark | 样本数 | 类型 | 难度 | 说明 |
|-----------|--------|------|------|------|
| **mini_benchmarks** | **180** | 混合 | ⭐⭐ | **快速评测首选** (~10 分钟) |

包含：
- `chinese_basic`: 30 题（中文指令遵循）
- `gsm8k_mini`: 50 题（数学推理）
- `arc_easy_mini`: 50 题（科学问答）
- `hellaswag_mini`: 50 题（常识推理）

### 全量版

| Benchmark | 样本数 | 类型 | 难度 | 说明 |
|-----------|--------|------|------|------|
| **chinese_basic** | 30 | 混合题型 | ⭐ 简单 | 中文基础题，适合弱模型评测 |
| **GSM8K** | 200 | 数学推理 | ⭐⭐⭐ 中等 | 小学数学应用题 |
| **Hellaswag** | 200 | 常识推理 | ⭐⭐ 中等 | 文本补全 |
| **ARC-Easy** | 570 | 科学问答 | ⭐⭐ 简单 | 小学科学选择题 |
| **ARC-Challenge** | 299 | 科学问答 | ⭐⭐⭐ 较难 | 中学科学选择题 |
| **TruthfulQA** | 100 | 事实性 | ⭐⭐⭐ 较难 | 抗幻觉测试 |
| **MMLU** | 198 | 综合知识 | ⭐⭐⭐⭐ 难 | 高中地理（子集） |
| **总计** | **1,597** | - | - | - |

## 使用方法

### 快速评测（推荐）⭐

```bash
# 快速评测（180 题，约 10 分钟）
python scripts/eval_benchmark_local.py \
    --benchmark mini \
    --adapter ./outputs

# 查看结果
cat ./outputs/benchmark_eval/summary_mini.json
```

### 评测单个 Benchmark

```bash
# 中文基础评测（30 题，约 2 分钟）
python scripts/eval_benchmark_local.py \
    --benchmark chinese_basic \
    --adapter ./outputs

# 数学推理评测（50 题，约 5 分钟）
python scripts/eval_benchmark_local.py \
    --benchmark gsm8k_mini \
    --adapter ./outputs

# 科学问答评测（50 题，约 5 分钟）
python scripts/eval_benchmark_local.py \
    --benchmark arc_easy_mini \
    --adapter ./outputs
```

### 全量评测

```bash
# 全量评测（约 1-2 小时）
python scripts/eval_benchmark_local.py \
    --benchmark all \
    --adapter ./outputs
```

## 数据格式

每个文件为 JSONL 格式，每行一个样本：

```json
{
  "benchmark": "gsm8k",
  "category": "math",
  "type": "math_word_problem",
  "prompt": "问题文本...",
  "answer": "正确答案",
  "choices": ["选项 A", "选项 B", "选项 C", "选项 D"]
}
```

## 使用方法

### 1. 直接推理测试

```python
import json
from src.evaluator import SFTEvaluator

# 加载评测数据
with open("bench_test_data/gsm8k.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# 使用 evaluator 进行测试
evaluator = SFTEvaluator(model, tokenizer)

correct = 0
for item in data[:50]:  # 测试前 50 题
    response = evaluator.generate(item["prompt"])
    # 提取答案并比较
    # ...
```

### 2. 使用评测脚本

```bash
# 评测单个 benchmark
python scripts/benchmark_eval.py \
    --base_model ./models/qwen2.5-0.5b-instruct \
    --adapter ./outputs \
    --tasks gsm8k \
    --limit 50

# 评测多个 benchmark
python scripts/benchmark_eval.py \
    --tasks arc,hellaswag,truthfulqa
```

### 3. 本地评估准确率

```bash
python scripts/eval_benchmark_local.py \
    --benchmark gsm8k \
    --adapter ./outputs
```

## 评估指标建议

| Benchmark | 评估指标 |
|-----------|----------|
| GSM8K | Exact Match (精确匹配数字答案) |
| ARC | Accuracy (选择题准确率) |
| Hellaswag | Accuracy (选择题准确率) |
| TruthfulQA | Accuracy (事实正确率) |
| MMLU | Accuracy (选择题准确率) |

## 重新下载数据

如需重新下载或调整采样比例：

```bash
python scripts/download_benchmark_data.py
```

## 注意事项

1. **中文数据集**：C-Eval 和 CMMLU 由于格式问题无法直接下载，建议使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 进行评测
2. **MMLU**：由于数据集太大，这里只包含高中地理一个子集
3. **完整评测**：如需完整评测，请使用 `lm_eval --model hf --tasks mmlu,ceval,cmmlu...`

## 参考链接

- [GSM8K](https://huggingface.co/datasets/gsm8k)
- [Hellaswag](https://huggingface.co/datasets/hellaswag)
- [ARC](https://huggingface.co/datasets/ai2_arc)
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa)
- [MMLU](https://huggingface.co/datasets/cais/mmlu)
- [C-Eval](https://huggingface.co/datasets/ceval/ceval-valid)
- [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu)
