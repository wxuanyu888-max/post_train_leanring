# 模型效果测试指南

本指南介绍如何测试和对比基座模型（微调前）与微调后模型的效果。

## 测试流程

### 第一步：测试基座模型效果

在微调之前，先测试基座模型的原始效果：

```bash
# 演示模式 - 测试各种类型的任务
python scripts/test_base_model.py --mode demo

# 数据模式 - 在真实数据上测试
python scripts/test_base_model.py \
    --mode data \
    --data_path ./data/raw/alpaca-500.jsonl \
    --max_samples 20 \
    --output ./outputs/base_model_results.json

# 交互式模式 - 手动测试
python scripts/test_base_model.py --mode interactive
```

### 第二步：训练模型

```bash
# 数据预处理
python scripts/preprocess_data.py

# 开始训练
python scripts/train.py --use_4bit
```

### 第三步：对比微调前后效果

```bash
# 对比基座模型和微调模型
python scripts/compare_models.py \
    --base_model ./models/qwen2.5-0.5b-instruct \
    --adapter ./outputs \
    --data_path ./data/raw/alpaca-500.jsonl \
    --max_samples 10 \
    --output ./outputs/comparison_results.json
```

## 测试用例示例

### 1. 知识问答

**Instruction**: What is machine learning?

**期望**: 解释机器学习的定义和基本概念

**基座模型可能的输出**:
```
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without being explicitly programmed. It uses algorithms to
identify patterns in data and make predictions or decisions.
```

### 2. 数学计算

**Instruction**: Solve: 15 + 27 * 3 = ?

**期望**: 正确的计算结果 (96)

**测试点**: 基座模型可能计算错误，微调后应该更准确

### 3. 代码生成

**Instruction**: Write a Python function to calculate factorial

**期望**:
```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

### 4. 翻译任务

**Instruction**: Translate to Chinese
**Input**: Hello, how are you today?

**期望**: 你好，你今天好吗？

### 5. 文本创作

**Instruction**: Write a short poem about spring

**期望**: 一首关于春天的短诗

## 评估指标

### 定性评估

| 维度 | 评分标准 |
|------|----------|
| 准确性 | 回答是否正确 |
| 相关性 | 是否切题 |
| 完整性 | 回答是否完整 |
| 流畅性 | 语言是否通顺 |
| 格式 | 是否符合要求（如代码格式） |

### 定量指标

```python
# 从对比结果中计算
{
    "total_samples": 10,
    "avg_base_length": 45.3,      # 基座模型平均输出长度
    "avg_ft_length": 78.6,        # 微调模型平均输出长度
    "improvement_rate": 0.73,     # 改进比例（人工评估）
}
```

## 预期效果对比

### Qwen2.5-0.5B 基座模型特点

- **优势**:
  - 基本的语言理解能力
  - 简单的问答可以应对
  - 有一定的知识储备

- **劣势**:
  - 输出可能较短
  - 复杂指令遵循能力弱
  - 代码生成能力有限
  - 可能产生幻觉内容

### 微调后预期改进

- **指令遵循**: 更好地理解并遵循复杂指令
- **输出质量**: 回答更详细、更有条理
- **领域适配**: 在特定领域（如客服）表现更好
- **格式规范**: 更好地遵循输出格式要求

## 完整测试报告示例

运行以下命令生成完整测试报告：

```bash
# 1. 测试基座模型
python scripts/test_base_model.py \
    --mode data \
    --max_samples 20 \
    --output ./outputs/base_results.json

# 2. 训练模型
python scripts/train.py

# 3. 对比效果
python scripts/compare_models.py \
    --max_samples 20 \
    --output ./outputs/comparison.json

# 4. 查看结果
cat ./outputs/comparison.json
```

## 注意事项

1. **温度参数**: 较低温度 (0.3-0.5) 输出更确定，较高温度 (0.7-1.0) 更有创造性
2. **最大长度**: 根据任务类型调整 `max_new_tokens`
3. **样本选择**: 确保测试样本覆盖各种任务类型
4. **人工评估**: 自动指标有限，建议人工抽查评估

## 常见问题

**Q: 微调后效果不如基座模型？**
A: 可能原因：
- 训练数据质量差
- 训练轮次过多导致过拟合
- 学习率过大
- 数据格式不正确

**Q: 输出都是乱码？**
A: 检查：
- tokenizer 是否正确加载
- 提示模板格式是否正确
- pad_token 是否设置

**Q: 显存不足怎么办？**
A: 使用 QLoRA 模式：
```bash
python scripts/train.py --use_4bit
```
