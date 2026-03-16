# 使用示例

## 1. 数据预处理

### 基本用法
```bash
python scripts/preprocess_data.py
```

### 自定义参数
```bash
python scripts/preprocess_data.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --test_split 0.2 \
    --min_length 20 \
    --max_samples 5000
```

### 参数说明
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dir` | ./data/raw | 原始数据目录 |
| `--output_dir` | ./data/processed | 输出目录 |
| `--test_split` | 0.1 | 验证集比例 |
| `--min_length` | 10 | 最小输出长度过滤 |
| `--max_samples` | None | 最大样本数限制 |


## 2. 训练

### 标准 LoRA 训练
```bash
python scripts/train.py
```

### QLoRA 训练 (节省显存)
```bash
python scripts/train.py --use_4bit
```

### 自定义配置
```bash
python scripts/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config.yaml \
    --output_dir ./my_experiment \
    --train_data ./data/processed/train.jsonl \
    --valid_data ./data/processed/valid.jsonl
```

### 修改训练参数 (临时覆盖配置文件)
```bash
# 使用更小的 batch size 和更多 epoch
cat > configs/training_config_override.yaml << EOF
training:
  num_train_epochs: 5
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 1.0e-4
EOF

python scripts/train.py --training_config configs/training_config_override.yaml
```


## 3. 评估

### 基本评估
```bash
python scripts/evaluate.py
```

### 自定义参数
```bash
python scripts/evaluate.py \
    --base_model ./models/qwen2.5-0.5b-instruct \
    --adapter ./outputs \
    --eval_data ./data/processed/valid.jsonl \
    --output ./outputs/eval_results.json \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --max_samples 100
```


## 4. 推理

### 交互式对话
```bash
python scripts/inference.py --interactive
```

### 单次推理
```bash
python scripts/inference.py \
    --instruction "Explain what is machine learning" \
    --temperature 0.7 \
    --max_new_tokens 256
```

### 带输入的推理
```bash
python scripts/inference.py \
    --instruction "Translate the following text to Chinese" \
    --input "Hello, how are you?" \
    --temperature 0.5
```


## 5. 使用 Python API

### 训练
```python
from src.model import load_tokenizer, load_base_model, create_lora_config, apply_lora
from src.dataset import load_sft_dataset
from src.trainer import create_trainer, SFTTrainingConfig

# 加载组件
tokenizer = load_tokenizer("./models/qwen2.5-0.5b-instruct")
model = load_base_model("./models/qwen2.5-0.5b-instruct")

# 应用 LoRA
lora_config = create_lora_config(r=8, alpha=16)
model = apply_lora(model, lora_config)

# 加载数据
train_dataset = load_sft_dataset(
    "./data/processed/train.jsonl",
    tokenizer,
    max_length=512,
)

# 创建训练器
trainer = create_trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    config=SFTTrainingConfig(
        output_dir="./outputs",
        num_train_epochs=3,
        per_device_train_batch_size=4,
    ),
)

# 训练
trainer.train()
trainer.save_model("./outputs")
```

### 推理
```python
from src.inference import SFTInference

# 加载模型
inference = SFTInference(
    base_model_path="./models/qwen2.5-0.5b-instruct",
    adapter_path="./outputs",
)

# 生成响应
response = inference.generate(
    instruction="Write a poem about spring",
    max_new_tokens=200,
    temperature=0.8,
)
print(response)

# 交互式对话
inference.interactive_chat()
```

### 评估
```python
from src.evaluator import SFTEvaluator
import json

# 加载评估数据
with open("./data/processed/valid.jsonl") as f:
    eval_data = [json.loads(line) for line in f]

# 创建评估器
evaluator = SFTEvaluator(model, tokenizer)

# 评估
results = evaluator.evaluate_dataset(
    eval_data,
    output_path="./eval_results.json",
    max_new_tokens=256,
)

# 计算指标
metrics = evaluator.compute_metrics(results["results"])
print(metrics)
```


## 6. 显存优化技巧

### 显存占用对比

| 模式 | 显存占用 | 速度 |
|------|----------|------|
| 全精度 | ~3GB | 最快 |
| FP16 混合精度 | ~2GB | 快 |
| 8bit 量化 | ~1.5GB | 中等 |
| 4bit 量化 (QLoRA) | ~1GB | 较慢 |

### 优化方法

```bash
# 方法 1: 使用 QLoRA
python scripts/train.py --use_4bit

# 方法 2: 减小 batch size，增加梯度累积
# 修改 configs/training_config.yaml:
# per_device_train_batch_size: 1
# gradient_accumulation_steps: 16

# 方法 3: 减小序列长度
# max_seq_length: 256

# 方法 4: 启用梯度检查点
# gradient_checkpointing: true
```


## 7. 多数据集混合训练

```python
from src.dataset import merge_datasets, create_train_valid_split

# 合并多个数据集
dataset = merge_datasets(
    data_paths=[
        "./data/raw/alpaca-cleaned.jsonl",
        "./data/raw/belle-500.jsonl",
        "./data/raw/customer-service-100.jsonl",
    ],
    weights=[1.0, 1.0, 2.0],  # customer_service 采样 2 倍
)

# 划分训练集和验证集
train_dataset, valid_dataset = create_train_valid_split(
    dataset,
    valid_ratio=0.1,
    seed=42,
)
```
