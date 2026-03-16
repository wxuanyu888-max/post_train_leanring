"""
DPO 数据集加载与预处理模块
用于加载偏好数据集 (chosen/rejected 对)
"""
import json
from typing import Dict, List, Optional, Union
from pathlib import Path

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import PreTrainedTokenizer


class DPODataset:
    """DPO 偏好数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None,
    ):
        """
        初始化 DPO 数据集

        Args:
            data_path: 数据文件路径 (JSONL 格式)
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示模板
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._default_template()
        self.data_path = data_path

    def _default_template(self) -> str:
        """默认提示模板"""
        return "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"

    def format_prompt(self, example: Dict) -> Dict:
        """
        格式化单个 DPO 样本

        Args:
            example: 包含 instruction, input, chosen, rejected 的字典

        Returns:
            格式化后的字典，包含 prompt, chosen, rejected
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        chosen = example.get("chosen", "")
        rejected = example.get("rejected", "")

        # 构建 prompt
        prompt = self.prompt_template.format(
            instruction=instruction,
            input=input_text
        )

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    def __call__(self, examples: Union[Dict, List[Dict]]) -> Dict:
        """
        处理批量数据

        Args:
            examples: 单个样本或样本列表

        Returns:
            格式化后的字典
        """
        if isinstance(examples, dict):
            examples = [examples]

        return [self.format_prompt(ex) for ex in examples]


def load_dpo_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    split: str = "train",
    prompt_template: Optional[str] = None,
) -> Dataset:
    """
    加载 DPO 数据集

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        split: 数据集划分名称
        prompt_template: 提示模板

    Returns:
        处理后的 Dataset 对象

    Note:
        DPO 数据格式 (JSONL):
        {
            "instruction": "用户指令",
            "input": "可选的输入内容",
            "chosen": "优选的回复",
            "rejected": "拒绝的回复"
        }
    """
    # 加载 JSONL 数据集
    dataset = load_dataset("json", data_files=data_path, split=split)

    # 设置默认模板
    template = prompt_template or "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"

    def format_function(example):
        """格式化函数"""
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
        format_function,
        desc="Formatting DPO prompts",
        num_proc=4,
    )

    return dataset


def convert_sft_to_dpo_format(
    sft_dataset: Dataset,
    rejected_model_output: str = "",
) -> Dataset:
    """
    将 SFT 数据集转换为 DPO 格式

    Args:
        sft_dataset: SFT 格式的数据集
        rejected_model_output: 拒绝样本的来源

    Returns:
        DPO 格式的数据集

    Note:
        此函数用于将已有的 SFT 数据转换为 DPO 格式
        需要有 chosen 和 rejected 的区分
    """
    def convert(example):
        return {
            "prompt": example.get("instruction", "") + " " + example.get("input", ""),
            "chosen": example.get("output", ""),
            "rejected": rejected_model_output or "抱歉，我无法回答这个问题。",
        }

    dataset = sft_dataset.map(
        convert,
        desc="Converting to DPO format",
        num_proc=4,
    )

    return dataset


def merge_dpo_datasets(
    data_paths: List[str],
    weights: Optional[List[float]] = None,
    seed: int = 42,
) -> Dataset:
    """
    合并多个 DPO 数据集

    Args:
        data_paths: 数据文件路径列表
        weights: 每个数据集的采样权重
        seed: 随机种子

    Returns:
        合并后的 Dataset 对象
    """
    datasets = []
    for path in data_paths:
        ds = load_dataset("json", data_files=path, split="train")
        datasets.append(ds)

    if weights:
        # 根据权重采样
        sampled_datasets = []
        for ds, weight in zip(datasets, weights):
            num_samples = int(len(ds) * weight)
            sampled = ds.select(range(min(num_samples, len(ds))))
            sampled_datasets.append(sampled)
        datasets = sampled_datasets

    merged = concatenate_datasets(datasets)
    merged = merged.shuffle(seed=seed)

    return merged


def create_dpo_train_valid_split(
    dataset: Dataset,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    创建 DPO 训练集和验证集划分

    Args:
        dataset: 原始数据集
        valid_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (train_dataset, valid_dataset) 元组
    """
    dataset = dataset.shuffle(seed=seed)
    split = dataset.train_test_split(test_size=valid_ratio, seed=seed)
    return split["train"], split["test"]


def filter_dpo_dataset(
    dataset: Dataset,
    min_chosen_length: int = 10,
    max_chosen_length: int = 2048,
    min_rejected_length: int = 10,
    max_rejected_length: int = 2048,
) -> Dataset:
    """
    过滤 DPO 数据集

    Args:
        dataset: DPO 数据集
        min_chosen_length: chosen 最小长度
        max_chosen_length: chosen 最大长度
        min_rejected_length: rejected 最小长度
        max_rejected_length: rejected 最大长度

    Returns:
        过滤后的数据集
    """
    def filter_fn(example):
        chosen_len = len(example.get("chosen", ""))
        rejected_len = len(example.get("rejected", ""))

        return (
            min_chosen_length <= chosen_len <= max_chosen_length and
            min_rejected_length <= rejected_len <= max_rejected_length
        )

    filtered = dataset.filter(
        filter_fn,
        desc="Filtering DPO dataset",
        num_proc=4,
    )

    return filtered
