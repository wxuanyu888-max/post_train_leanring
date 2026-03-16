# SFT - Supervised Fine-Tuning

基于 **HuggingFace TRL + LoRA** 的大语言模型微调项目，用于学习/研究目的。

## 环境要求

### 使用现有 Conda 环境

如果您的 conda 环境已有 torch 和 transformers（如 `d2l` 环境），可以直接使用：

```bash
# 激活环境
conda activate d2l

# 验证安装
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### 安装缺失依赖

如果缺少部分依赖，可以补充安装：

```bash
conda activate d2l
pip install datasets peft trl accelerate
```

## 项目结构

```
sft/
├── configs/                 # 配置文件
│   ├── model_config.yaml    # 模型和 LoRA 配置
│   ├── training_config.yaml # 训练配置
│   └── data_config.yaml     # 数据配置
├── src/                     # 核心模块
│   ├── __init__.py
│   ├── dataset.py           # 数据集加载与预处理
│   ├── model.py             # 模型加载与 LoRA 配置
│   ├── trainer.py           # 训练器封装
│   ├── evaluator.py         # 评估模块
│   └── inference.py         # 推理模块
├── scripts/                 # 可执行脚本
│   ├── preprocess_data.py   # 数据预处理
│   ├── train.py             # 训练启动
│   ├── evaluate.py          # 模型评估
│   ├── inference.py         # 推理脚本
│   ├── benchmark_eval.py    # 使用 lm-eval 进行基准评测
│   ├── eval_benchmark_local.py  # 本地 benchmark 评测
│   ├── download_benchmark_data.py  # 下载 benchmark 数据
│   ├── test_base_model.py   # 基座模型测试
│   ├── compare_models.py    # 模型对比工具
│   └── demo.py              # 快速效果演示
├── bench_test_data/         # Benchmark 评测数据
│   ├── mini_benchmarks.jsonl    # 精简版合集 (180 题，推荐)
│   ├── chinese_basic.jsonl      # 中文基础 (30 题)
│   ├── gsm8k_mini.jsonl         # 数学推理精简版 (50 题)
│   ├── arc_easy_mini.jsonl      # 科学问答精简版 (50 题)
│   ├── hellaswag_mini.jsonl     # 常识推理精简版 (50 题)
│   ├── gsm8k.jsonl              # 数学推理完整版 (200 题)
│   ├── arc_easy.jsonl           # 科学问答完整版 (570 题)
│   ├── arc_challenge.jsonl      # 科学问答 - 挑战 (299 题)
│   ├── hellaswag.jsonl          # 常识推理完整版 (200 题)
│   ├── truthfulqa.jsonl         # 事实性评测 (100 题)
│   ├── mmlu.jsonl               # 综合知识 (198 题)
│   └── combined_benchmarks.jsonl  # 合并数据 (1567 题)
├── data/
│   ├── raw/                 # 原始数据
│   └── processed/           # 处理后数据
├── models/
│   ├── qwen2.5-0.5b-instruct/  # 基座模型
│   └── checkpoints/         # 训练检查点
├── outputs/                 # 训练输出和日志
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 激活 Conda 环境

```bash
conda activate d2l
```

### 2. 准备数据

将 JSONL 格式的数据文件放入 `data/raw/` 目录：

```bash
# 示例数据格式 (每行一个 JSON 对象)
{"instruction": "指令", "input": "输入", "output": "期望输出"}
```

### 3. 数据预处理

```bash
conda activate d2l
python scripts/preprocess_data.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --test_split 0.1
```

### 4. 测试基座模型效果

```bash
# 快速演示
python scripts/demo.py

# 详细测试
python scripts/test_base_model.py --mode demo
```

### 5. 开始训练

```bash
# 标准 LoRA 训练
python scripts/train.py

# QLoRA 训练 (节省显存，需要 bitsandbytes)
python scripts/train.py --use_4bit
```

### 6. 对比微调前后效果

```bash
python scripts/compare_models.py \
    --base_model ./models/qwen2.5-0.5b-instruct \
    --adapter ./outputs \
    --max_samples 10
```

### 7. 快速评测（推荐）

```bash
# 快速评测（180 题，约 10 分钟）
python scripts/eval_benchmark_local.py \
    --benchmark mini \
    --adapter ./outputs

# 查看单个 benchmark 结果
python scripts/eval_benchmark_local.py \
    --benchmark chinese_basic \
    --adapter ./outputs
```

### 8. 模型推理

```bash
# 交互式对话
python scripts/inference.py --interactive

# 单次推理
python scripts/inference.py \
    --instruction "Explain what machine learning is" \
    --temperature 0.7
```

## 配置说明

### LoRA 配置 (`configs/model_config.yaml`)

```yaml
lora:
  r: 8                    # LoRA 秩，越大参数越多
  alpha: 16               # 通常为 2*r
  dropout: 0.05           # Dropout 比例
  target_modules:         # 目标模块
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### 训练配置 (`configs/training_config.yaml`)

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4    # 等效 batch_size = 16
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  fp16: true                        # 如果 GPU 支持
  gradient_checkpointing: true      # 节省显存
```

## 显存优化

如果显存不足：

1. **启用 QLoRA (4bit 量化)**:
   ```bash
   python scripts/train.py --use_4bit
   ```

2. **减小 batch size**:
   ```yaml
   per_device_train_batch_size: 1
   gradient_accumulation_steps: 16
   ```

3. **减小序列长度**:
   ```yaml
   max_seq_length: 256
   ```

## 核心模块说明

| 模块 | 说明 |
|------|------|
| `dataset.py` | 数据集加载、格式化、预处理 |
| `model.py` | 模型加载、LoRA 配置、量化支持 |
| `trainer.py` | SFTTrainer 封装、训练配置 |
| `evaluator.py` | 模型评估、指标计算 |
| `inference.py` | 模型推理、交互式对话 |
| `test_base_model.py` | 基座模型效果测试 |
| `compare_models.py` | 微调前后效果对比 |

## 支持的模型

本项目以 Qwen2.5-0.5B-Instruct 为例，理论上支持任何 HuggingFace 模型：

- Qwen / Qwen2 / Qwen2.5
- Llama / Llama2 / Llama3
- Mistral
- Baichuan
- Yi

只需修改 `configs/model_config.yaml` 中的 `name_or_path` 即可。

## 许可证

MIT License
