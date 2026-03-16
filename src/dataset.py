"""
数据集加载与预处理模块
"""
import json
from typing import Dict, List, Optional, Union
from pathlib import Path

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


class SFTDataset:
    """监督微调数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_template: Optional[str] = None,
    ):
        """
        初始化 SFT 数据集

        Args:
            data_path: 数据文件路径 (JSONL 格式)
            tokenizer: 分词器
            max_length: 最大序列长度
            prompt_template: 提示模板，支持 {instruction}, {input}, {output} 占位符
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template or self._default_template()
        self.data_path = data_path

    def _default_template(self) -> str:
        """默认提示模板"""
        return "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n{output}"

    def format_prompt(self, example: Dict) -> str:
        """
        格式化单个样本

        Args:
            example: 包含 instruction, input, output 的字典

        Returns:
            格式化后的完整提示文本
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        prompt = self.prompt_template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
        return prompt

    def __call__(self, examples: Union[Dict, List[Dict]]) -> Dict:
        """
        处理批量数据

        Args:
            examples: 单个样本或样本列表

        Returns:
            分词后的字典，包含 input_ids, attention_mask, labels
        """
        if isinstance(examples, dict):
            examples = [examples]

        texts = [self.format_prompt(ex) for ex in examples]

        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )

        # labels 用于计算损失，与 input_ids 相同
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized


def load_sft_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    split: str = "train",
    prompt_template: Optional[str] = None,
) -> Dataset:
    """
    加载 SFT 数据集

    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        split: 数据集划分名称
        prompt_template: 提示模板

    Returns:
        处理后的 Dataset 对象
    """
    # 加载 JSONL 数据集
    dataset = load_dataset("json", data_files=data_path, split=split)

    # 设置默认模板
    template = prompt_template or "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n{output}"

    def format_function(example):
        """格式化函数"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        prompt = template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )

        return {"text": prompt}

    # 应用格式化
    dataset = dataset.map(
        format_function,
        desc="Formatting prompts",
        num_proc=4,
    )

    return dataset


def merge_datasets(
    data_paths: List[str],
    weights: Optional[List[float]] = None,
    seed: int = 42,
) -> Dataset:
    """
    合并多个数据集

    Args:
        data_paths: 数据文件路径列表
        weights: 每个数据集的采样权重
        seed: 随机种子

    Returns:
        合并后的 Dataset 对象
    """
    from datasets import concatenate_datasets

    datasets = []
    for path in data_paths:
        ds = load_dataset("json", data_files=path, split="train")
        datasets.append(ds)

    if weights:
        # 根据权重采样
        from datasets import Dataset
        sampled_datasets = []
        for ds, weight in zip(datasets, weights):
            num_samples = int(len(ds) * weight)
            sampled = ds.select(range(min(num_samples, len(ds))))
            sampled_datasets.append(sampled)
        datasets = sampled_datasets

    merged = concatenate_datasets(datasets)
    merged = merged.shuffle(seed=seed)

    return merged


def create_train_valid_split(
    dataset: Dataset,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """
    创建训练集和验证集划分

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
