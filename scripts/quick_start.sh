#!/bin/bash
# SFT 项目快速启动脚本 (使用 conda d2l 环境)

set -e

echo "=============================================="
echo "SFT Quick Start (Conda d2l environment)"
echo "=============================================="

# 激活 conda 环境
source $(which conda)/../etc/profile.d/conda.sh
conda activate d2l

echo ""
echo "Conda environment: d2l"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# 1. 数据预处理
echo ""
echo "Step 1: Preprocessing data..."
python scripts/preprocess_data.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --test_split 0.1

# 2. 开始训练
echo ""
echo "Step 2: Starting training..."
python scripts/train.py \
    --model_config configs/model_config.yaml \
    --training_config configs/training_config.yaml \
    --output_dir ./outputs

echo ""
echo "=============================================="
echo "Quick Start completed!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  Evaluate: python scripts/evaluate.py"
echo "  Inference: python scripts/inference.py --interactive"
