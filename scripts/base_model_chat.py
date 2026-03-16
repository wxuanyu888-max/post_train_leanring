#!/usr/bin/env python3
"""
基座模型交互式对话脚本
在终端直接与 Qwen2.5-0.5B-Instruct 对话
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str):
    """加载模型"""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def chat(model, tokenizer):
    """交互式对话"""
    print("\n" + "=" * 70)
    print(" Qwen2.5-0.5B-Instruct 基座模型 - 交互式对话 ")
    print("=" * 70)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空对话历史")
    print("=" * 70 + "\n")

    conversation_history = []

    while True:
        try:
            # 获取用户输入
            user_input = input("\033[1;32mYou:\033[0m ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                conversation_history = []
                print("Conversation history cleared.")
                continue

            if not user_input:
                continue

            # 构建提示
            prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"

            # 分词
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 生成
            print("\033[1;34mAssistant:\033[0m ", end="", flush=True)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.5,
                    top_p=0.85,
                    top_k=40,
                    repetition_penalty=1.2,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

            print(response)
            print()

            # 保存对话历史
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="基座模型交互式对话")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/qwen2.5-0.5b-instruct",
        help="模型路径",
    )
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model(args.model_path)

    # 开始对话
    chat(model, tokenizer)


if __name__ == "__main__":
    main()
