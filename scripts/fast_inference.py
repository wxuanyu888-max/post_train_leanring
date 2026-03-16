#!/usr/bin/env python3
"""
快速推理脚本 - 优化版
启用流式输出，提升感知速度
"""
import argparse
import sys
import time

from src.inference import SFTInference


def parse_args():
    parser = argparse.ArgumentParser(description="Fast SFT Inference with Streaming")

    parser.add_argument(
        "--base_model",
        type=str,
        default="./models/qwen2.5-0.5b-instruct",
        help="基座模型路径",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./outputs",
        help="LoRA Adapter 路径",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="指令",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="输入文本 (可选)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成 token 数 (不限制输出)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k 采样",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p 采样",
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="禁用流式输出",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="显示生成速度统计",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Fast SFT Inference (Streaming Enabled)")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print("=" * 60)

    # 加载模型
    print("\nLoading model...")
    inference = SFTInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter if args.adapter else None,
    )

    # 推理
    if args.no_stream:
        # 普通模式
        start = time.time()
        response = inference.generate(
            instruction=args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        elapsed = time.time() - start

        print(f"\nResponse: {response}")

        if args.timing:
            print(f"\nGeneration time: {elapsed:.2f}s")
            print(f"Speed: ~{len(response) / elapsed:.1f} chars/sec")
    else:
        # 流式模式
        print("\n[Response]: ", end="", flush=True)

        start = time.time()
        token_count = 0
        for token in inference.stream_generate(
            instruction=args.instruction,
            input_text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ):
            print(token, end="", flush=True)
            token_count += 1

        elapsed = time.time() - start

        print()  # 换行

        if args.timing:
            print(f"\n--- Statistics ---")
            print(f"Tokens generated: {token_count}")
            print(f"Generation time: {elapsed:.2f}s")
            print(f"Speed: {token_count / elapsed:.1f} tokens/sec")


if __name__ == "__main__":
    main()
