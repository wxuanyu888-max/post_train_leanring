#!/bin/bash

# Post Train Learning - 快速启动脚本

echo "=== Post Train Learning ==="
echo "1. 安装依赖"
echo "2. 下载模型"
echo "3. 训练 SFT"
echo "4. 训练 DPO"
echo "5. 模型评测"
echo "6. 退出"
echo ""

read -p "请选择 [1-6]: " choice

case $choice in
    1)
        echo "安装依赖..."
        pip install -r requirements.txt
        ;;
    2)
        echo "下载模型 (Qwen2.5-0.5B)..."
        git lfs install
        git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct ./models/qwen2.5-0.5b-instruct
        ;;
    3)
        echo "训练 SFT..."
        python scripts/train.py
        ;;
    4)
        echo "训练 DPO..."
        python scripts/train_dpo.py
        ;;
    5)
        echo "模型评测 (180 题)..."
        python scripts/eval_benchmark_local.py --benchmark mini
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac
