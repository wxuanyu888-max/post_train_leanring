# DPO Training Guide

## DPO 训练快速开始

### 1. 数据准备

DPO 数据需要包含偏好对 (chosen/rejected)：

```jsonl
{
  "instruction": "用户指令",
  "input": "可选的输入",
  "chosen": "优选的回复",
  "rejected": "拒绝的回复"
}
```

### 2. 配置文件

- `configs/dpo_model_config.yaml` - 模型配置
- `configs/dpo_training_config.yaml` - 训练配置

### 3. 启动训练

```bash
# 基础训练
python scripts/train_dpo.py

# 指定配置
python scripts/train_dpo.py \
  --model_config configs/dpo_model_config.yaml \
  --training_config configs/dpo_training_config.yaml \
  --train_data ./data/dpo/train.jsonl \
  --output_dir ./outputs/dpo

# 使用量化
python scripts/train_dpo.py --use_8bit

# 调整 DPO 参数
python scripts/train_dpo.py --beta 0.5 --loss_type sigmoid
```

## DPO 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beta` | 0.1 | 温度参数，控制偏离参考模型的程度，越大越保守 |
| `loss_type` | sigmoid | 损失类型：sigmoid, ipo, kto_pair |

## DPO vs SFT

| 特性 | SFT | DPO |
|------|-----|-----|
| 数据格式 | instruction/input/output | instruction/input/chosen/rejected |
| 训练目标 | 最大化似然 | 偏好优化 |
| 参考模型 | 不需要 | 需要（自动创建副本） |
| 学习率 | 1e-4 ~ 1e-5 | 5e-7 ~ 1e-6（更小） |

## 典型训练流程

1. **SFT 预训练**: 先用 SFT 训练基座模型
2. **DPO 微调**: 在 SFT 模型基础上用 DPO 优化偏好

```bash
# Step 1: SFT
python scripts/train.py --output_dir ./outputs/sft

# Step 2: 更新 dpo_model_config.yaml 中的 name_or_path 为 SFT 输出
# Step 3: DPO
python scripts/train_dpo.py --output_dir ./outputs/dpo
```

## 数据结构示例

参见 `data/dpo/train_example.jsonl`
