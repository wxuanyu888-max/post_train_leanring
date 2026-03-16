"""
训练器封装模块
基于 HuggingFace TRL 的 SFTTrainer
"""
from dataclasses import dataclass, field
from typing import Optional, List
import os
import json
from pathlib import Path
from datetime import datetime

from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from src.logger import (
    get_callbacks,
    setup_logger,
    generate_training_report,
    StructuredLoggingCallback,
    TensorBoardCallback,
)


@dataclass
class SFTTrainingConfig:
    """
    训练配置数据类

    Example:
        config = SFTTrainingConfig(
            output_dir="./outputs",
            num_train_epochs=3,
            per_device_train_batch_size=4,
        )
    """
    # 输出配置
    output_dir: str = "./outputs"
    report_to: str = "none"  # 不报告到 wandb 等
    resume_from_checkpoint: str = None  # 从 checkpoint 继续训练

    # 训练轮次和批次
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # 学习率和调度
    learning_rate: float = 2.0e-4
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
    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    # 优化器和精度
    optim: str = "paged_adamw_8bit"
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True

    # 数据配置
    max_seq_length: int = 512
    dataset_num_proc: int = 4
    packing: bool = False  # 是否打包多个样本

    # 其他
    seed: int = 42
    dataloader_num_workers: int = 4

    # 实验跟踪
    run_name: Optional[str] = None

    # 日志配置
    use_tensorboard: bool = True  # 是否启用 TensorBoard
    log_level: str = "info"  # 日志级别：debug, info, warning


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset=None,
    config: Optional[SFTTrainingConfig] = None,
) -> SFTTrainer:
    """
    创建 SFT 训练器

    Args:
        model: 模型 (已应用 LoRA)
        tokenizer: 分词器
        train_dataset: 训练数据集
        eval_dataset: 验证数据集 (可选)
        config: 训练配置

    Returns:
        SFTTrainer 对象
    """
    if config is None:
        config = SFTTrainingConfig()

    # 检测设备
    import torch
    use_mps = torch.backends.mps.is_available() and not (
        "Intel" in __import__("subprocess").check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode()
    )
    use_accelerate = torch.cuda.is_available() or use_mps

    # 创建训练参数
    training_args = SFTConfig(
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
        eval_strategy=config.evaluation_strategy,  # 新版参数名
        eval_steps=config.eval_steps,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        use_mps_device=use_mps,
        use_cpu=not use_accelerate,
        resume_from_checkpoint=config.resume_from_checkpoint,
    )
    device_name = "mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_name}")

    # 定义格式化函数 - 将数据格式化为完整的 instruction 格式
    def format_example(example):
        """
        格式化单个样本为完整的 instruction 格式

        Args:
            example: 包含 instruction, input, output 的字典

        Returns:
            格式化后的完整文本
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # 构建完整格式
        if input_text and input_text.strip():
            full_text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Output:\n{output}"
            )
        else:
            full_text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Output:\n{output}"
            )
        return full_text

    # 获取回调函数
    callbacks = get_callbacks(
        output_dir=config.output_dir,
        use_tensorboard=config.use_tensorboard
    )

    # 创建训练器 - 使用 formatting_func 代替 dataset_text_field
    # 这样 TRL 能更好地处理 prompt 和 output 的分离
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # 新版 API 用 processing_class 代替 tokenizer
        formatting_func=format_example,  # 使用格式化函数
    )

    return trainer


def create_training_args_from_dict(args_dict: dict) -> SFTConfig:
    """
    从字典创建训练参数

    Args:
        args_dict: 包含训练参数的字典

    Returns:
        SFTConfig 对象
    """
    return SFTConfig(**args_dict)


class TrainerCallback:
    """
    训练回调基类

    Example:
        class LossCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    print(f"Step {state.global_step}: loss = {logs.get('loss', 0):.4f}")
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时调用"""
        pass

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时调用"""
        pass

    def on_epoch_end(self, args, state, control, **kwargs):
        """每个 epoch 结束时调用"""
        pass

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        """每个 step 结束时调用"""
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        """日志记录时调用"""
        pass

    def on_save(self, args, state, control, **kwargs):
        """保存 checkpoint 时调用"""
        pass

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """评估时调用"""
        pass


class TrainingProgressTracker:
    """
    训练进度跟踪器
    用于记录和可视化训练进度
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics_history: List[dict] = []
        self.start_time = None

    def log(self, step: int, metrics: dict):
        """记录训练指标"""
        entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)

    def save(self):
        """保存进度记录"""
        output_file = self.output_dir / "training_progress.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)

    def get_summary(self) -> dict:
        """获取训练摘要"""
        if not self.metrics_history:
            return {}

        losses = [m.get("loss", 0) for m in self.metrics_history if "loss" in m]
        return {
            "total_steps": len(self.metrics_history),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "avg_loss": sum(losses) / len(losses) if losses else None,
        }
