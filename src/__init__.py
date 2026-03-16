"""
SFT - Supervised Fine-Tuning for LLM
基于 HuggingFace TRL + LoRA 的大语言模型微调项目
"""

from src.dataset import SFTDataset, load_sft_dataset
from src.model import (
    load_tokenizer,
    load_base_model,
    create_lora_config,
    apply_lora,
    print_trainable_params,
)
from src.trainer import create_trainer, SFTTrainingConfig
from src.evaluator import SFTEvaluator
from src.inference import SFTInference

# DPO - Direct Preference Optimization
from src.dataset_dpo import DPODataset, load_dpo_dataset, filter_dpo_dataset
from src.trainer_dpo import create_dpo_trainer, DPOTrainingConfig, prepare_dpo_dataset

__version__ = "1.0.0"
__all__ = [
    # SFT
    "SFTDataset",
    "load_sft_dataset",
    "load_tokenizer",
    "load_base_model",
    "create_lora_config",
    "apply_lora",
    "print_trainable_params",
    "create_trainer",
    "SFTTrainingConfig",
    "SFTEvaluator",
    "SFTInference",
    # DPO
    "DPODataset",
    "load_dpo_dataset",
    "filter_dpo_dataset",
    "create_dpo_trainer",
    "DPOTrainingConfig",
    "prepare_dpo_dataset",
]
