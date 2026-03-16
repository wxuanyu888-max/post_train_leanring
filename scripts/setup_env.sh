#!/bin/bash
# SFT 项目环境设置脚本

echo "=============================================="
echo "SFT Environment Setup"
echo "=============================================="

D2L_PYTHON="/opt/miniconda3/envs/d2l/bin/python"
D2L_PIP="/opt/miniconda3/envs/d2l/bin/pip"

# 检查 d2l 环境是否存在
if [ ! -f "$D2L_PYTHON" ]; then
    echo "Error: d2l conda environment not found!"
    exit 1
fi

echo "Using d2l environment: $D2L_PYTHON"
echo ""

# 显示当前已安装的包
echo "Checking installed packages..."
$D2L_PYTHON -c "
try:
    import torch
    print(f'  torch: {torch.__version__}')
except:
    print('  torch: not installed')

try:
    import transformers
    print(f'  transformers: {transformers.__version__}')
except:
    print('  transformers: not installed')

try:
    import datasets
    print(f'  datasets: {datasets.__version__}')
except:
    print('  datasets: not installed')
"
echo ""

# 安装缺失的依赖
echo "Installing missing packages..."
$D2L_PIP install transformers datasets accelerate peft trl pyyaml scikit-learn tqdm

echo ""
echo "Verifying installation..."
$D2L_PYTHON -c "
import torch
import transformers
import datasets
import peft
import trl
print(f'OK: PyTorch {torch.__version__}')
print(f'OK: Transformers {transformers.__version__}')
print(f'OK: Datasets {datasets.__version__}')
print(f'OK: PEFT {peft.__version__}')
print(f'OK: TRL {trl.__version__}')
"

echo ""
echo "=============================================="
echo "Setup completed!"
echo "=============================================="
echo ""
echo "Now run:"
echo "  $D2L_PYTHON scripts/demo.py"
