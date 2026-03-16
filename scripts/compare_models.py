#!/usr/bin/env python3
"""
微调效果对比脚本
对比基座模型和微调后模型的效果
"""
import torch
import json
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class ModelComparator:
    """模型对比器"""

    def __init__(
        self,
        base_model_path: str,
        adapter_path: Optional[str] = None,
    ):
        """
        初始化对比器

        Args:
            base_model_path: 基座模型路径
            adapter_path: LoRA Adapter 路径
        """
        print("Loading base model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if adapter_path and Path(adapter_path).exists():
            print(f"Loading adapter from: {adapter_path}")
            self.finetuned_model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
            )
        else:
            print("No adapter found, using base model for both")
            self.finetuned_model = self.base_model

        self.base_model.eval()
        self.finetuned_model.eval()

    def generate(
        self,
        model,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """生成响应"""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def compare_single(
        self,
        instruction: str,
        input_text: str = "",
        reference: str = "",
    ) -> Dict:
        """对比单个样本"""
        print(f"\nInstruction: {instruction}")
        if input_text:
            print(f"Input: {input_text}")
        if reference:
            print(f"Reference: {reference[:100]}...")

        base_response = self.generate(self.base_model, instruction, input_text)
        ft_response = self.generate(self.finetuned_model, instruction, input_text)

        print("-" * 60)
        print(f"Base Model:     {base_response[:150]}...")
        print(f"Finetuned Model: {ft_response[:150]}...")

        return {
            "instruction": instruction,
            "input": input_text,
            "reference": reference,
            "base_model_output": base_response,
            "finetuned_model_output": ft_response,
        }

    def compare_dataset(
        self,
        data_path: str,
        max_samples: int = 10,
        output_path: Optional[str] = None,
    ) -> Dict:
        """在数据集上对比"""
        print(f"\nLoading data from: {data_path}")

        test_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                test_data.append(json.loads(line))

        print(f"Testing {len(test_data)} samples...")

        results = []
        for i, sample in enumerate(test_data, 1):
            print(f"\n[{i}/{len(test_data)}]")

            result = self.compare_single(
                instruction=sample.get("instruction", ""),
                input_text=sample.get("input", ""),
                reference=sample.get("output", ""),
            )
            results.append(result)

        # 保存结果
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_path}")

        return {
            "results": results,
            "count": len(results),
        }


def print_comparison_report(results: Dict):
    """打印对比报告"""
    print("\n" + "=" * 70)
    print("COMPARISON REPORT")
    print("=" * 70)

    for i, result in enumerate(results, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Instruction: {result['instruction'][:60]}...")
        print(f"\n[Reference]")
        print(f"{result['reference'][:200]}...")

        print(f"\n[Base Model Output]")
        print(f"{result['base_model_output'][:200]}...")

        print(f"\n[Finetuned Model Output]")
        print(f"{result['finetuned_model_output'][:200]}...")

    print("\n" + "=" * 70)


def compute_basic_metrics(results: Dict) -> Dict:
    """计算基础指标"""
    metrics = {
        "total_samples": len(results),
        "avg_base_length": 0,
        "avg_ft_length": 0,
    }

    total_base = 0
    total_ft = 0

    for result in results:
        total_base += len(result["base_model_output"].split())
        total_ft += len(result["finetuned_model_output"].split())

    metrics["avg_base_length"] = total_base / len(results)
    metrics["avg_ft_length"] = total_ft / len(results)

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare base and finetuned models")
    parser.add_argument(
        "--base_model",
        type=str,
        default="./models/qwen2.5-0.5b-instruct",
        help="Base model path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./outputs",
        help="LoRA adapter path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/raw/alpaca-500.jsonl",
        help="Test data path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum samples to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/comparison_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive comparison mode",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MODEL COMPARISON TOOL")
    print("=" * 70)
    print(f"Base Model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print("=" * 70)

    # 创建对比器
    comparator = ModelComparator(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )

    if args.interactive:
        # 交互式对比
        print("\nINTERACTIVE MODE")
        print("Type 'quit' to exit")

        while True:
            instruction = input("\n[Instruction] > ").strip()
            if instruction.lower() in ["quit", "exit", "q"]:
                break

            input_text = input("[Input] (optional) > ").strip()

            result = comparator.compare_single(instruction, input_text)

    else:
        # 数据集对比
        results = comparator.compare_dataset(
            data_path=args.data_path,
            max_samples=args.max_samples,
            output_path=args.output,
        )

        # 打印报告
        print_comparison_report(results["results"])

        # 计算指标
        metrics = compute_basic_metrics(results["results"])
        print("\nMETRICS:")
        print(f"  Total samples: {metrics['total_samples']}")
        print(f"  Avg base model output length: {metrics['avg_base_length']:.1f} words")
        print(f"  Avg finetuned model output length: {metrics['avg_ft_length']:.1f} words")


if __name__ == "__main__":
    main()
