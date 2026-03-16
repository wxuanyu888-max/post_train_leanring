"""
评估模块
"""
import torch
from typing import List, Dict, Optional
from pathlib import Path
import json

from tqdm import tqdm


class SFTEvaluator:
    """
    SFT 模型评估器

    Example:
        evaluator = SFTEvaluator(model, tokenizer)
        results = evaluator.evaluate_dataset(eval_data)
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        初始化评估器

        Args:
            model: 模型对象
            tokenizer: 分词器
            max_length: 最大输入长度
            device: 运行设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> str:
        """
        生成响应

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_p: Top-p 采样参数
            top_k: Top-k 采样参数
            do_sample: 是否采样

        Returns:
            生成的文本
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # 解码时跳过输入部分
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response

    def evaluate_single(
        self,
        instruction: str,
        input_text: str = "",
        reference: str = "",
        **generate_kwargs,
    ) -> Dict:
        """
        评估单个样本

        Args:
            instruction: 指令
            input_text: 输入文本
            reference: 参考输出
            **generate_kwargs: 生成参数

        Returns:
            包含生成结果和指标的字典
        """
        prompt = self._format_prompt(instruction, input_text)
        generated = self.generate(prompt, **generate_kwargs)

        result = {
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "generated": generated,
        }

        return result

    def evaluate_dataset(
        self,
        eval_data: List[Dict],
        output_path: Optional[str] = None,
        **generate_kwargs,
    ) -> Dict:
        """
        评估整个数据集

        Args:
            eval_data: 评估数据列表
            output_path: 结果保存路径
            **generate_kwargs: 生成参数

        Returns:
            评估结果字典
        """
        results = []

        for example in tqdm(eval_data, desc="Evaluating"):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            reference = example.get("output", "")

            result = self.evaluate_single(
                instruction=instruction,
                input_text=input_text,
                reference=reference,
                **generate_kwargs,
            )

            results.append(result)

        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return {"results": results, "count": len(results)}

    def _format_prompt(
        self,
        instruction: str,
        input_text: str = "",
    ) -> str:
        """格式化提示"""
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n"

    def compute_metrics(
        self,
        results: List[Dict],
    ) -> Dict:
        """
        计算评估指标

        Args:
            results: 评估结果列表

        Returns:
            指标字典

        Note:
            目前返回基础统计信息，可扩展 BLEU、ROUGE 等指标
        """
        metrics = {
            "total_samples": len(results),
            "avg_generated_length": 0,
            "avg_reference_length": 0,
        }

        total_generated = 0
        total_reference = 0

        for result in results:
            total_generated += len(result["generated"].split())
            total_reference += len(result["reference"].split())

        metrics["avg_generated_length"] = total_generated / len(results)
        metrics["avg_reference_length"] = total_reference / len(results)

        return metrics
