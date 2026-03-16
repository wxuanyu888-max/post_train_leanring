#!/usr/bin/env python3
"""
DPO 训练启动脚本
基于 HuggingFace TRL + LoRA 的 DPO 训练
"""
import argparse
import os
from pathlib import Path

import torch
import yaml

from src.model import (
    load_tokenizer,
    load_base_model,
    create_lora_config,
    apply_lora,
    print_trainable_params,
    print_lora_info,
)
from peft import PeftModel
from src.dataset_dpo import load_dpo_dataset, filter_dpo_dataset
from src.trainer_dpo import create_dpo_trainer, DPOTrainingConfig
from src.logger import generate_training_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train DPO model with LoRA")

    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="模型配置文件路径",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default="configs/dpo_training_config.yaml",
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录 (覆盖配置文件)",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="./data/dpo/train.jsonl",
        help="训练数据路径",
    )
    parser.add_argument(
        "--valid_data",
        type=str,
        default="./data/dpo/valid.jsonl",
        help="验证数据路径",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="使用 4bit 量化 (QLoRA)",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="使用 8bit 量化",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO 温度参数",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "ipo", "kto_pair"],
        help="DPO 损失类型",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="从已有的 LoRA adapter 继续训练",
    )

    return parser.parse_args()


def load_yaml_config(path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def print_training_info(config: dict, model_config: dict):
    """打印训练信息"""
    print("\n" + "=" * 60)
    print("DPO Training Configuration")
    print("=" * 60)

    print(f"\nModel: {model_config['model']['name_or_path']}")
    print(f"Model dtype: {model_config['model']['torch_dtype']}")

    print(f"\nLoRA Config:")
    print(f"  r: {model_config['lora']['r']}")
    print(f"  alpha: {model_config['lora']['alpha']}")
    print(f"  dropout: {model_config['lora']['dropout']}")
    print(f"  target_modules: {model_config['lora']['target_modules']}")

    print(f"\nDPO Config:")
    print(f"  beta: {config['dpo']['beta']}")
    print(f"  loss_type: {config['dpo']['loss_type']}")

    print(f"\nTraining Config:")
    print(f"  epochs: {config['training']['num_train_epochs']}")
    print(f"  batch_size: {config['training']['per_device_train_batch_size']}")
    print(f"  gradient_accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  learning_rate: {config['training']['learning_rate']}")
    print(f"  max_seq_length: {config['data']['max_seq_length']}")

    print(f"\nOptimization:")
    print(f"  fp16: {config['training']['fp16']}")
    print(f"  gradient_checkpointing: {config['training']['gradient_checkpointing']}")
    print(f"  optim: {config['training']['optim']}")

    print("=" * 60 + "\n")


def main():
    args = parse_args()

    # 加载配置
    print("Loading configurations...")
    model_config = load_yaml_config(args.model_config)
    training_config = load_yaml_config(args.training_config)

    # 覆盖命令行参数
    if args.beta:
        training_config["dpo"]["beta"] = args.beta
    if args.loss_type:
        training_config["dpo"]["loss_type"] = args.loss_type

    # 打印训练信息
    print_training_info(training_config, model_config)

    # 确定输出目录
    output_dir = args.output_dir or training_config["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存配置到输出目录
    with open(Path(output_dir) / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f, default_flow_style=False)
    with open(Path(output_dir) / "training_config.yaml", "w") as f:
        yaml.dump(training_config, f, default_flow_style=False)

    # 加载分词器
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(
        model_path=model_config["model"]["name_or_path"],
        padding_side="right",  # DPO 用 right padding
    )
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # 确定数据类型
    torch_dtype = getattr(torch, model_config["model"]["torch_dtype"])

    # 加载基座模型
    print("Loading base model...")
    model = load_base_model(
        model_path=model_config["model"]["name_or_path"],
        use_4bit=args.use_4bit or model_config.get("quantization", {}).get("use_4bit", False),
        use_8bit=args.use_8bit,
        torch_dtype=torch_dtype,
    )

    # 创建 LoRA 配置
    print("Creating LoRA config...")
    lora_config = create_lora_config(
        r=model_config["lora"]["r"],
        alpha=model_config["lora"]["alpha"],
        dropout=model_config["lora"]["dropout"],
        target_modules=model_config["lora"]["target_modules"],
    )

    # 应用 LoRA
    print("Applying LoRA...")
    model = apply_lora(model, lora_config)

    # 如果指定了 adapter，则加载已有 LoRA 继续训练
    if args.adapter and Path(args.adapter).exists():
        print(f"Loading existing LoRA adapter from: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        print("Loaded existing adapter, will continue training from this checkpoint")

    # 打印 LoRA 信息
    print_lora_info(model)

    # 加载训练数据集
    print("Loading DPO training dataset...")
    train_dataset = load_dpo_dataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=training_config["data"]["max_seq_length"],
    )

    # 过滤数据集
    print("Filtering dataset...")
    train_dataset = filter_dpo_dataset(
        train_dataset,
        min_chosen_length=10,
        max_chosen_length=training_config["data"]["max_seq_length"],
        min_rejected_length=10,
    )
    print(f"Train dataset size after filtering: {len(train_dataset)}")

    # 加载验证集 (如果存在)
    eval_dataset = None
    if Path(args.valid_data).exists():
        print("Loading DPO validation dataset...")
        eval_dataset = load_dpo_dataset(
            data_path=args.valid_data,
            tokenizer=tokenizer,
            max_length=training_config["data"]["max_seq_length"],
        )
        eval_dataset = filter_dpo_dataset(eval_dataset)
        print(f"Valid dataset size: {len(eval_dataset)}")

    # 创建 DPO 训练器
    print("Creating DPO trainer...")
    trainer = create_dpo_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        ref_model=None,  # 使用 model 的副本作为参考模型
        config=DPOTrainingConfig(
            output_dir=output_dir,
            num_train_epochs=training_config["training"]["num_train_epochs"],
            per_device_train_batch_size=training_config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=training_config["training"]["gradient_accumulation_steps"],
            learning_rate=training_config["training"]["learning_rate"],
            warmup_ratio=training_config["training"]["warmup_ratio"],
            lr_scheduler_type=training_config["training"]["lr_scheduler_type"],
            logging_steps=training_config["training"]["logging_steps"],
            save_steps=training_config["training"]["save_steps"],
            save_total_limit=training_config["training"]["save_total_limit"],
            fp16=training_config["training"]["fp16"],
            bf16=training_config["training"]["bf16"],
            gradient_checkpointing=training_config["training"]["gradient_checkpointing"],
            optim=training_config["training"]["optim"],
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=training_config["training"].get("eval_steps", 100) if eval_dataset else None,
            # DPO 特有参数
            beta=training_config["dpo"]["beta"],
            loss_type=training_config["dpo"]["loss_type"],
        ),
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("Starting DPO training...")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=None)

    # 生成训练报告
    print("\nGenerating training report...")
    report = generate_training_report(output_dir)
    print(report)

    # 保存模型
    print("\nSaving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 保存训练日志
    trainer.state.save_to_json(Path(output_dir) / "trainer_state.json")

    print("\n" + "=" * 60)
    print("DPO training completed!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60 + "\n")

    # 打印显存使用情况
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()
