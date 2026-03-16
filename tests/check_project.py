#!/usr/bin/env python3
"""
项目结构验证脚本
检查所有必需的文件和依赖是否正确
"""
import sys
from pathlib import Path


def check_file(path: str, name: str) -> bool:
    """检查文件是否存在"""
    exists = Path(path).exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")
    return exists


def check_import(module: str) -> bool:
    """检查模块是否可以导入"""
    try:
        __import__(module)
        print(f"  ✓ {module}")
        return True
    except ImportError as e:
        print(f"  ✗ {module}: {e}")
        return False


def main():
    print("=" * 60)
    print("SFT Project Structure Verification")
    print("=" * 60)

    all_passed = True

    # 检查目录结构
    print("\n1. Checking directory structure...")
    directories = [
        "configs",
        "src",
        "scripts",
        "data/raw",
        "data/processed",
        "outputs",
        "models/qwen2.5-0.5b-instruct",
    ]

    for dir_path in directories:
        exists = Path(dir_path).is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_passed = False
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"      -> Created directory")

    # 检查配置文件
    print("\n2. Checking configuration files...")
    config_files = [
        ("configs/model_config.yaml", "Model config"),
        ("configs/training_config.yaml", "Training config"),
        ("configs/data_config.yaml", "Data config"),
    ]

    for path, name in config_files:
        if not check_file(path, name):
            all_passed = False

    # 检查源代码文件
    print("\n3. Checking source files...")
    source_files = [
        ("src/__init__.py", "Package init"),
        ("src/dataset.py", "Dataset module"),
        ("src/model.py", "Model module"),
        ("src/trainer.py", "Trainer module"),
        ("src/evaluator.py", "Evaluator module"),
        ("src/inference.py", "Inference module"),
    ]

    for path, name in source_files:
        if not check_file(path, name):
            all_passed = False

    # 检查脚本文件
    print("\n4. Checking scripts...")
    script_files = [
        ("scripts/preprocess_data.py", "Preprocess script"),
        ("scripts/train.py", "Train script"),
        ("scripts/evaluate.py", "Evaluate script"),
        ("scripts/inference.py", "Inference script"),
    ]

    for path, name in script_files:
        if not check_file(path, name):
            all_passed = False

    # 检查其他文件
    print("\n5. Checking other files...")
    other_files = [
        ("requirements.txt", "Requirements"),
        ("README.md", "README"),
        ("USAGE_EXAMPLES.md", "Usage examples"),
    ]

    for path, name in other_files:
        if not check_file(path, name):
            all_passed = False

    # 检查数据文件
    print("\n6. Checking data files...")
    data_files = [
        ("data/raw/alpaca-500.jsonl", "Alpaca 500"),
        ("data/raw/alpaca-cleaned.jsonl", "Alpaca cleaned"),
        ("data/raw/belle-500.jsonl", "Belle 500"),
        ("data/raw/customer-service-100.jsonl", "Customer service"),
    ]

    for path, name in data_files:
        check_file(path, name)

    # 检查模型文件
    print("\n7. Checking model files...")
    model_files = [
        ("models/qwen2.5-0.5b-instruct/config.json", "Model config"),
        ("models/qwen2.5-0.5b-instruct/tokenizer_config.json", "Tokenizer config"),
    ]

    for path, name in model_files:
        check_file(path, name)

    # 检查 Python 依赖
    print("\n8. Checking Python dependencies...")
    dependencies = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "trl",
        "sklearn",
        "yaml",
    ]

    for module in dependencies:
        if not check_import(module):
            all_passed = False

    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed!")
    else:
        print("✗ Some checks failed. Please review the output above.")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
