"""
DPO 训练器封装模块
基于 HuggingFace TRL 的 DPOTrainer
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import json
from pathlib import Path
from datetime import datetime

from transformers import TrainingArguments
from trl import DPOTrainer, DPOConfig
from src.logger import (
    get_callbacks,
    setup_logger,
    generate_training_report,
    StructuredLoggingCallback,
    TensorBoardCallback,
)


@dataclass
class DPOTrainingConfig:
    """
    DPO 训练配置数据类

    Example:
        config = DPOTrainingConfig(
            output_dir="./outputs/dpo",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            beta=0.1,
        )
    """
    # 输出配置
    output_dir: str = "./outputs/dpo"
    report_to: str = "none"  # 不报告到 wandb 等

    # 训练轮次和批次
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # 学习率和调度
    learning_rate: float = 5.0e-7  # DPO 通常用更小的学习率
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # 日志和保存
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    save_strategy: str = "steps"
    logging_strategy: str = "steps"

    # 评估配置
    evaluation_strategy: str = "no"
    eval_steps: int = 100

    # 优化器和精度
    optim: str = "adamw_torch"
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = True

    # DPO 特有参数
    beta: float = 0.1  # DPO 的温度参数，控制偏离参考模型的程度
    loss_type: str = "sigmoid"  # 损失类型：sigmoid, ipo, kto_pair

    # 其他
    seed: int = 42
    dataloader_num_workers: int = 4

    # 实验跟踪
    run_name: Optional[str] = None

    # 日志配置
    use_tensorboard: bool = True
    log_level: str = "info"

    # 参考模型配置
    use_reference_model: bool = True  # 是否使用参考模型


def create_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    ref_model=None,
    config: Optional[DPOTrainingConfig] = None,
) -> DPOTrainer:
    """
    创建 DPO 训练器

    Args:
        model: 模型 (已应用 LoRA)
        tokenizer: 分词器
        train_dataset: 训练数据集 (需包含 chosen/rejected 字段)
        eval_dataset: 验证数据集 (可选)
        ref_model: 参考模型，如果为 None 则使用 model 的副本
        config: 训练配置

    Returns:
        DPOTrainer 对象
    """
    if config is None:
        config = DPOTrainingConfig()

    # 检测设备
    import torch
    use_mps = torch.backends.mps.is_available() and not (
        "Intel" in __import__("subprocess").check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
    )
    use_accelerate = torch.cuda.is_available() or use_mps

    # 创建训练参数
    training_args = DPOConfig(
        output_dir=config.output_dir,
        report_to=config.report_to,
        run_name=config.run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_strategy=config.save_strategy,
        logging_strategy=config.logging_strategy,
        eval_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        use_mps_device=use_mps,
        use_cpu=not use_accelerate,
        # DPO 特有参数
        beta=config.beta,
        loss_type=config.loss_type,
    )
    device_name = "mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_name}")

    # 获取回调函数
    callbacks = get_callbacks(
        output_dir=config.output_dir,
        use_tensorboard=config.use_tensorboard
    )

    # 创建 DPO 训练器
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    return trainer


def create_dpo_training_args_from_dict(args_dict: dict) -> DPOConfig:
    """
    从字典创建训练参数

    Args:
        args_dict: 包含训练参数的字典

    Returns:
        DPOConfig 对象
    """
    return DPOConfig(**args_dict)


def prepare_dpo_dataset(
    dataset,
    tokenizer,
    max_length: int = 512,
    prompt_template: Optional[str] = None,
) -> Any:
    """
    准备 DPO 数据集格式

    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大序列长度
        prompt_template: 提示模板

    Returns:
        处理后的数据集

    Note:
        DPO 数据集需要包含以下字段:
        - prompt: 提示文本
        - chosen: 优选回复
        - rejected: 拒绝回复
    """
    template = prompt_template or "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n"

    def format_dpo_example(example: Dict) -> Dict:
        """格式化 DPO 样本"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        # 构建 prompt
        prompt = template.format(
            instruction=instruction,
            input=input_text
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # 应用格式化
    dataset = dataset.map(
        format_dpo_example,
        desc="Formatting DPO prompts",
        num_proc=4,
    )

    return dataset
