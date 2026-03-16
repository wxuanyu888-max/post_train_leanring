#!/usr/bin/env python3
"""
推理脚本
使用微调后的模型进行推理
"""
import argparse
import sys

from src.inference import SFTInference


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Model Inference")

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
        "--interactive",
        action="store_true",
        help="交互式对话模式",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="指令 (非交互模式)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="输入文本 (非交互模式)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,  # 不限制输出长度
        help="最大生成 token 数",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="温度参数",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用流式输出 (实时显示生成内容)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k 采样参数",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p 采样参数",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SFT Model Inference")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print("=" * 60)

    # 加载模型
    print("\nLoading model...")
    inference = SFTInference(
        base_model_path=args.base_model,
        adapter_path=args.adapter if args.adapter else None,
    )

    if args.interactive:
        # 交互式对话（流式输出）
        print("\n")
        inference.interactive_chat()

    else:
        # 单次推理
        instruction = args.instruction
        if instruction is None:
            # 从标准输入读取
            print("\nEnter instruction (or 'quit' to exit):")
            instruction = input("> ").strip()
            if instruction.lower() in ["quit", "exit", "q"]:
                sys.exit(0)
            input_text = input("Enter input (optional): ").strip()
        else:
            input_text = args.input

        if args.stream:
            # 流式输出模式
            print("\n[Response]: ", end="", flush=True)
            for token in inference.stream_generate(
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            ):
                print(token, end="", flush=True)
            print()  # 换行
        else:
            # 普通模式
            response = inference.generate(
                instruction=instruction,
                input_text=input_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            print("\n" + "=" * 60)
            print(f"Instruction: {instruction}")
            if input_text:
                print(f"Input: {input_text}")
            print(f"\nResponse: {response}")
            print("=" * 60)


if __name__ == "__main__":
    main()
