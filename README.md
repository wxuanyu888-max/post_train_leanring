# Post Train Learning

基于 HuggingFace TRL + LoRA 的大语言模型微调项目，支持 SFT 监督微调和 DPO 偏好优化。

## 功能特性

- **SFT 监督微调** - 基于指令数据微调模型
- **DPO 偏好优化** - 直接偏好优化对齐人类偏好
- **LoRA 高效微调** - 低秩适配器，减少显存占用
- **Benchmark 评测** - 内置 180 题快速评测

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- 8GB+ 显存（推荐）

## 快速开始

### 1. 克隆项目

```bash
git clone <repo_url>
cd post_train_learning
```

### 2. 一键启动

```bash
# 运行交互式启动脚本
bash scripts/quick_start.sh
```

或手动执行：

```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型（必做！）
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct ./models/qwen2.5-0.5b-instruct

# 训练 SFT
python scripts/train.py

# 训练 DPO（可选）
python scripts/train_dpo.py

# 模型评测
python scripts/eval_benchmark_local.py --benchmark mini
```

## 参数配置说明

### 模型选择

修改 `configs/model_config.yaml`：

```yaml
model:
  # 基座模型路径（支持 HuggingFace 模型）
  name_or_path: ./models/qwen2.5-0.5b-instruct
  model_type: qwen2
  trust_remote_code: true
```

### LoRA 配置

修改 `configs/model_config.yaml`：

```yaml
lora:
  # LoRA 秩，越大参数越多，效果越好，但训练更慢
  r: 8

  # LoRA alpha 参数，通常设为 r 的 2 倍
  alpha: 16

  # Dropout 比例，防止过拟合
  dropout: 0.05

  # 要训练的模块（Qwen2 适用）
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

### 训练配置

修改 `configs/training_config.yaml`：

```yaml
training:
  # 输出目录
  output_dir: ./outputs

  # 训练轮数
  num_train_epochs: 1

  # 批处理大小
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4  # 等效 batch_size = 16

  # 学习率
  learning_rate: 1.0e-4

  # 保存间隔
  save_steps: 100

data:
  # 训练数据路径
  train_file: ./data/sft-1000.jsonl

  # 验证集路径
  validation_file: ./data/processed/valid.jsonl

  # 最大序列长度
  max_seq_length: 512
```

### 评测配置

```bash
# 评测脚本
python scripts/eval_benchmark_local.py \
    --benchmark mini              # 快速评测（180 题）
    --adapter ./outputs/lora      # LoRA 权重路径

# 自定义评测数据
python scripts/eval_benchmark_local.py \
    --benchmark mini \
    --data_path ./data/bench_test_data/mini_benchmarks.jsonl
```

## 目录结构

```
post_train_learning/
├── src/                    # 核心代码
│   ├── __init__.py
│   ├── model.py           # 模型加载
│   ├── dataset.py         # SFT 数据集
│   ├── dataset_dpo.py     # DPO 数据集
│   ├── trainer.py         # SFT 训练器
│   ├── trainer_dpo.py    # DPO 训练器
│   ├── evaluator.py      # 评估器
│   ├── inference.py      # 推理
│   ├── benchmark_compare.py  # 评测对比
│   └── logger.py         # 日志
├── scripts/               # 脚本
│   └── quick_start.sh    # 快速启动
├── configs/               # 配置文件
│   ├── model_config.yaml  # 模型/LoRA 配置
│   ├── training_config.yaml  # 训练配置
│   ├── dpo_model_config.yaml
│   └── dpo_training_config.yaml
├── data/                  # 数据
│   ├── sft-1000.jsonl    # SFT 训练数据
│   ├── alpaca-cleaned.jsonl
│   ├── belle-500.jsonl
│   ├── customer-service-100.jsonl
│   ├── dpo/              # DPO 训练数据
│   ├── bench_test_data/  # 评测数据
│   │   └── mini_benchmarks.jsonl  # 180 题
│   └── processed/        # 处理后的数据
├── outputs/               # 训练输出
│   ├── lora/            # LoRA 权重
│   └── log/             # 训练/评测日志
├── models/               # 模型目录（需下载）
├── docs/                 # 文档
├── tests/                # 测试
├── .gitignore
├── requirements.txt
└── README.md
```


## 训练结论

我们在 Qwen2.5-0.5B-Instruct 基座模型上进行了多种训练，使用 180 题 mini benchmarks 评测：

| 模型 | 训练数据 | LoRA | 正确数 | 准确率 | 相比基座提升 |
|------|----------|------|--------|--------|--------------|
| Baseline | - | ❌ | 31/180 | 17.22% | - |
| SFT-1000 | sft-1000.jsonl | ✅ | 38/180 | 21.11% | +3.89% |
| Alpaca1000 | alpaca-cleaned.jsonl | ✅ | 32/180 | 17.78% | +0.56% |
| SFT-V1 | sft-1000.jsonl | ✅ | 32/180 | 17.78% | +0.56% |
| DPO-V1 | dpo | ✅ | 37/180 | 20.56% | +3.34% |

![训练结果对比](./docs/training_results.png)

### 主要发现

1. **0.5B 小模型局限**：在复杂推理任务上表现有限，数学题几乎无法正确回答，最重要更本无法理解题目意思，题目里面有一个很直接的话直接给出答案，但是模型还是喜欢推理。
2. **SFT-1000 效果最好**：准确率达到 21.11%，相比基座提升 3.89 个百分点
3. **数据质量很重要**：只要训练相交基准模型都有提升，但是对于这种小模型来说很难把控，很难学习到训练的参数，并且并不是叠加的更好，每个的题目正确都是不同的，并不是一个线性提升，你训练过后的模型很可能把原来的题做错了。

### 问题分析

- **选择题**：模型表现较好，能正确回答约 20-30% 的选择题
- **数学计算题**：几乎无法正确提取答案，倾向于输出完整解题过程
- **开放性问答**：答案格式不统一，难以评估

## 常见问题

### Q: 模型文件太大，无法下载？

A: 使用 HuggingFace CLI 或镜像站点：
```bash
# 方法1: 使用 HF CLI
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# 方法2: 从其他镜像站点下载
```

### Q: 显存不足？

A: 修改配置降低显存占用：
```yaml
lora:
  r: 4  # 降低 LoRA 秩

training:
  per_device_train_batch_size: 2  # 减小 batch size
  gradient_checkpointing: true    # 启用梯度检查点
```

### Q: 如何使用自己的数据？

A: 将数据整理成对应格式，放入 `data/` 目录，修改配置文件中的 `train_file` 路径。

## 参考

- [HuggingFace TRL](https://huggingface.co/docs/trl)
- [LoRA](https://github.com/microsoft/LoRA)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
