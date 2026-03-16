#!/usr/bin/env python3
"""
CPU 批量推理脚本
一次处理多个样本，充分利用 CPU 并行能力
"""
import argparse
import json
import time
import torch
from pathlib import Path
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class BatchInference:
    """CPU 批量推理类"""

    def __init__(
        self,
        model_path: str,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        """
        初始化批量推理

        Args:
            model_path: 模型路径
            batch_size: 批量大小 (CPU 推荐 2-8)
            max_length: 最大输入长度
        """
        print(f"Loading model: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU 用 float32
            trust_remote_code=True,
        )

        self.model.eval()
        self.batch_size = batch_size
        self.max_length = max_length

        print(f"Model loaded. Batch size: {batch_size}")

    @torch.no_grad()
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
    ) -> List[str]:
        """
        批量生成 - 真正的并行处理

        Args:
            prompts: prompt 列表
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: Top-p
            top_k: Top-k

        Returns:
            生成的响应列表
        """
        # 分词（自动 padding 到 batch 内最长）
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        # 批量生成
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # KV Cache 加速
            num_beams=1,
        )

        # 解码每个样本的生成结果
        responses = []
        for i, output in enumerate(outputs):
            input_length = inputs["input_ids"][i].shape[0]
            generated_tokens = output[input_length:]
            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            responses.append(response)

        return responses

    def evaluate_dataset(
        self,
        data_path: str,
        output_path: str,
        max_samples: int = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict:
        """
        批量评估数据集

        Args:
            data_path: 数据文件路径 (JSONL)
            output_path: 输出文件路径
            max_samples: 最大样本数（None 表示全部）
            max_new_tokens: 最大生成 token 数
            temperature: 温度

        Returns:
            统计信息
        """
        # 加载数据
        print(f"Loading data from: {data_path}")
        data = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))

        if max_samples:
            data = data[:max_samples]

        print(f"Loaded {len(data)} samples")

        # 构建 prompts
        prompts = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Output:\n"
            prompts.append(prompt)

        # 批量推理
        all_results = []
        total_time = 0

        print(f"\nStarting batch inference (batch_size={self.batch_size})...")
        start_total = time.time()

        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Batches"):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_items = data[i : i + self.batch_size]

            batch_start = time.time()

            responses = self.batch_generate(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            batch_time = time.time() - batch_start
            total_time += batch_time

            # 保存结果
            for item, response in zip(batch_items, responses):
                all_results.append({
                    "instruction": item.get("instruction", ""),
                    "input": item.get("input", ""),
                    "reference": item.get("output", ""),
                    "generated": response,
                })

        total_time = time.time() - start_total

        # 保存结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # 统计信息
        stats = {
            "total_samples": len(all_results),
            "batch_size": self.batch_size,
            "total_time_sec": round(total_time, 2),
            "avg_time_per_sample": round(total_time / len(all_results), 3),
            "samples_per_second": round(len(all_results) / total_time, 2),
            "output_path": str(output_path),
        }

        print("\n" + "=" * 60)
        print("BATCH INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(all_results)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg time/sample: {stats['avg_time_per_sample']:.3f}s")
        print(f"Samples/sec: {stats['samples_per_second']:.2f}")
        print(f"Results saved to: {output_path}")
        print("=" * 60)

        return stats


def get_available_memory_gb() -> float:
    """获取可用内存 (GB)"""
    import platform
    system = platform.system()

    if system == "Linux":
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024 / 1024
    elif system == "Darwin":  # macOS
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True
        )
        total = int(result.stdout.strip()) / 1024 / 1024 / 1024
        # macOS 没有直接的 available 接口，估算 50% 可用
        return total * 0.5
    elif system == "Windows":
        import ctypes
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.wintypes.DWORD),
                ("dwMemoryLoad", ctypes.wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
            ]
        mem = MEMORYSTATUS()
        mem.dwLength = ctypes.sizeof(mem)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
        return mem.ullAvailPhys / 1024 / 1024 / 1024

    return 8.0  # 默认返回 8GB


def recommend_batch_size(available_memory_gb: float) -> int:
    """
    根据可用内存推荐 batch_size

    Qwen2.5-0.5B 估算：每个 batch 约 1.5GB
    """
    # 保留 2GB 给系统和其他进程
    safe_memory = max(0, available_memory_gb - 2)
    # 每个样本约 1.5GB
    recommended = int(safe_memory / 1.5)
    # 限制在 1-16 之间
    return max(1, min(16, recommended))


def main():
    parser = argparse.ArgumentParser(description="CPU Batch Inference")

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/qwen2.5-0.5b-instruct",
        help="模型路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="输入数据文件 (JSONL 格式)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/batch_results.json",
        help="输出结果文件",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批量大小 (不指定则自动根据内存推荐)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数 (用于测试)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数",
    )
    parser.add_argument(
        "--auto_batch",
        action="store_true",
        help="自动根据内存推荐 batch_size",
    )

    args = parser.parse_args()

    # 自动推荐 batch_size
    if args.batch_size is None or args.auto_batch:
        available = get_available_memory_gb()
        recommended = recommend_batch_size(available)
        print(f"Available memory: {available:.1f} GB")
        print(f"Recommended batch_size: {recommended}")
        args.batch_size = recommended

    # 创建推理器
    inferencer = BatchInference(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=512,
    )

    # 运行批量推理
    stats = inferencer.evaluate_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
