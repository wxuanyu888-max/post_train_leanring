# 训练数据分层说明

## 分层设计理念

参考 benchmark 的分层设计，训练数据也采用分层方案，支持快速实验到完整训练的不同需求。

---

## 数据分层结构

```
data/
├── raw/                           # 原始数据源
│   ├── alpaca-500.jsonl          # 500 条
│   ├── alpaca-cleaned.jsonl      # 51,760 条
│   ├── belle-500.jsonl           # 500 条
│   ├── customer-service-100.jsonl # 100 条
│   └── sft-1000.jsonl            # 1,000 条
│
├── train/                         # 分层训练数据（推荐使用）
│   ├── nano/                      # ⚡ 极速版 - 快速验证 (~50 条)
│   │   ├── train.jsonl           # 45 条训练
│   │   └── valid.jsonl           # 5 条验证
│   │
│   ├── mini/                      # 🚀 快速版 - 调参实验 (~500 条)
│   │   ├── train.jsonl           # 450 条训练
│   │   └── valid.jsonl           # 50 条验证
│   │
│   ├── small/                     # 📊 标准版 - 小型实验 (~2,000 条)
│   │   ├── train.jsonl           # 1,800 条训练
│   │   └── valid.jsonl           # 200 条验证
│   │
│   ├── medium/                    # 🎯 进阶版 - 中等训练 (~10,000 条)
│   │   ├── train.jsonl           # 9,000 条训练
│   │   └── valid.jsonl           # 1,000 条验证
│   │
│   └── large/                     # 🏆 完整版 - 生产训练 (~50,000+ 条)
│       ├── train.jsonl           # 45,000+ 条训练
│       └── valid.jsonl           # 5,000+ 条验证
│
└── processed/                     # 旧版兼容目录
    ├── train.jsonl
    └── valid.jsonl
```

---

## 各层级详细说明

### ⚡ nano 层 - 极速验证 (~50 条)

**用途**: 代码调试、流程验证、超参数快速测试

| 指标 | 数值 |
|------|------|
| 训练样本 | ~45 条 |
| 验证样本 | ~5 条 |
| 预计训练时间 | 1-3 分钟 |
| 显存需求 | < 4GB |
| 适用场景 | 验证训练流程、测试新配置 |

**数据来源**:
- alpaca-500: 20 条 (通用指令)
- sft-1000: 20 条 (综合任务)
- customer-service: 10 条 (对话场景)

---

### 🚀 mini 层 - 快速实验 (~500 条)

**用途**: 超参数调优、架构对比、快速迭代

| 指标 | 数值 |
|------|------|
| 训练样本 | ~450 条 |
| 验证样本 | ~50 条 |
| 预计训练时间 | 10-20 分钟 |
| 显存需求 | < 6GB |
| 适用场景 | 学习率/epochs 调优、LoRA 参数对比 |

**数据来源**:
- alpaca-500: 200 条
- belle-500: 150 条 (含中文)
- sft-1000: 100 条
- customer-service: 50 条

---

### 📊 small 层 - 标准实验 (~2,000 条)

**用途**: 正式实验、论文baseline、效果对比

| 指标 | 数值 |
|------|------|
| 训练样本 | ~1,800 条 |
| 验证样本 | ~200 条 |
| 预计训练时间 | 30-60 分钟 |
| 显存需求 | < 8GB |
| 适用场景 | 正式实验、模型对比、论文baseline |

**数据来源**:
- alpaca-500: 500 条 (全量)
- belle-500: 500 条 (全量)
- sft-1000: 800 条
- customer-service: 100 条 (全量)

---

### 🎯 medium 层 - 进阶训练 (~10,000 条)

**用途**: 生产环境测试、深度调优

| 指标 | 数值 |
|------|------|
| 训练样本 | ~9,000 条 |
| 验证样本 | ~1,000 条 |
| 预计训练时间 | 2-4 小时 |
| 显存需求 | < 12GB |
| 适用场景 | 生产环境验证、深度模型调优 |

**数据来源**:
- alpaca-cleaned: 8,000 条 (采样)
- belle-500: 500 条
- sft-1000: 1,000 条 (全量)

---

### 🏆 large 层 - 完整训练 (~50,000+ 条)

**用途**: 最终模型训练、生产部署

| 指标 | 数值 |
|------|------|
| 训练样本 | ~45,000+ 条 |
| 验证样本 | ~5,000+ 条 |
| 预计训练时间 | 8-12 小时 |
| 显存需求 | < 16GB (QLoRA) |
| 适用场景 | 最终模型训练、生产部署 |

**数据来源**:
- alpaca-cleaned: 51,760 条 (全量)
- belle-500: 500 条
- sft-1000: 1,000 条
- customer-service: 100 条

---

## 使用指南

### 快速开始

```bash
# 1. 流程验证 (nano 层)
python scripts/preprocess_data.py \
    --layer nano \
    --output_dir ./data/train/nano

# 2. 超参数调优 (mini 层)
python scripts/preprocess_data.py \
    --layer mini \
    --output_dir ./data/train/mini

# 3. 正式实验 (small 层)
python scripts/preprocess_data.py \
    --layer small \
    --output_dir ./data/train/small

# 4. 生产训练 (medium/large 层)
python scripts/preprocess_data.py \
    --layer medium \
    --output_dir ./data/train/medium
```

### 训练命令

```bash
# nano 层训练 - 验证流程
python scripts/train.py \
    --train_data ./data/train/nano/train.jsonl \
    --valid_data ./data/train/nano/valid.jsonl \
    --output_dir ./outputs/nano

# mini 层训练 - 调参实验
python scripts/train.py \
    --train_data ./data/train/mini/train.jsonl \
    --valid_data ./data/train/mini/valid.jsonl \
    --output_dir ./outputs/mini

# small 层训练 - 正式实验
python scripts/train.py \
    --train_data ./data/train/small/train.jsonl \
    --valid_data ./data/train/small/valid.jsonl \
    --output_dir ./outputs/small
```

---

## 数据分布统计

### 能力维度分布

| 层级 | 通用知识 | 数学推理 | 代码能力 | 语言理解 | 创意写作 |
|------|----------|----------|----------|----------|----------|
| nano | 20% | 20% | 15% | 25% | 20% |
| mini | 25% | 20% | 15% | 25% | 15% |
| small | 25% | 20% | 15% | 25% | 15% |
| medium | 30% | 20% | 15% | 20% | 15% |
| large | 35% | 20% | 15% | 15% | 15% |

### 语言分布

| 层级 | 英文 | 中文 | 多语言 |
|------|------|------|--------|
| nano | 70% | 25% | 5% |
| mini | 65% | 30% | 5% |
| small | 60% | 35% | 5% |
| medium | 60% | 35% | 5% |
| large | 55% | 40% | 5% |

---

## 与 Benchmark 对应关系

| 训练层级 | 推荐 Benchmark | 预计评测时间 |
|----------|----------------|--------------|
| nano | mini_benchmarks (180 题) | 5 分钟 |
| mini | mini_benchmarks (180 题) | 10 分钟 |
| small | chinese_basic + gsm8k_mini | 20 分钟 |
| medium | 完整 benchmark 子集 | 1 小时 |
| large | combined_benchmarks | 2-3 小时 |

---

## 最佳实践

### 推荐工作流

1. **开发阶段**: 使用 nano 层验证代码和流程
2. **调参阶段**: 使用 mini 层进行超参数搜索
3. **实验阶段**: 使用 small 层进行正式对比实验
4. **训练阶段**: 使用 medium/large 层训练最终模型

### 注意事项

- 各层级数据互不重叠，避免数据泄露
- 验证集从各数据源独立采样，确保分布一致
- 所有层级使用相同的随机种子 (seed=42)，保证可复现性
