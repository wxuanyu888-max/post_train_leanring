#!/usr/bin/env python3
"""
双 LoRA 叠加测试脚本
分别加载 SFT 和 DPO 的 LoRA 层，测试叠加效果
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_with_dual_lora(base_path, sft_path, dpo_path):
    """
    加载基座模型 + 双 LoRA 层

    Args:
        base_path: 基座模型路径
        sft_path: SFT LoRA 路径
        dpo_path: DPO LoRA 路径

    Returns:
        模型和分词器
    """
    print(f"Loading base model: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading SFT adapter: {sft_path}")
    model = PeftModel.from_pretrained(model, sft_path, adapter_name="sft")

    print(f"Loading DPO adapter: {dpo_path}")
    model.load_adapter(dpo_path, adapter_name="dpo")

    return model, tokenizer


def test_single_adapter(model, tokenizer, instruction, adapter_name):
    """测试单个 adapter"""
    model.set_adapter(adapter_name)
    prompt = f"### Instruction:\n{instruction}\n\n### Output:\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def test_dual_adapter(model, tokenizer, instruction, adapters=None):
    """测试双 adapter 叠加"""
    if adapters is None:
        adapters = ["sft", "dpo"]

    model.set_adapter(adapters)  # 同时激活多个
    prompt = f"### Instruction:\n{instruction}\n\n### Output:\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="Dual LoRA Adapter Test")

    parser.add_argument("--base_model", type=str, default="./models/qwen2.5-0.5b-instruct")
    parser.add_argument("--sft_adapter", type=str, default="./outputs/sft")
    parser.add_argument("--dpo_adapter", type=str, default="./outputs/dpo")
    parser.add_argument("--instruction", type=str, required=True, help="测试指令")

    args = parser.parse_args()

    print("=" * 70)
    print("Dual LoRA Adapter Test")
    print("=" * 70)

    # 加载模型
    model, tokenizer = load_model_with_dual_lora(
        args.base_model,
        args.sft_adapter,
        args.dpo_adapter,
    )

    print("\n" + "=" * 70)
    print(f"Instruction: {args.instruction}")
    print("=" * 70)

    # 测试 SFT 单独
    print("\n[1] SFT Only:")
    response_sft = test_single_adapter(model, tokenizer, args.instruction, "sft")
    print(response_sft)

    # 测试 DPO 单独
    print("\n[2] DPO Only:")
    response_dpo = test_single_adapter(model, tokenizer, args.instruction, "dpo")
    print(response_dpo)

    # 测试双 LoRA 叠加
    print("\n[3] SFT + DPO (Stacked):")
    response_dual = test_dual_adapter(model, tokenizer, args.instruction, ["sft", "dpo"])
    print(response_dual)

    # 测试加权组合（如果有这功能）
    print("\n[4] SFT + DPO (with weights):")
    # 可以手动调整 adapter 的缩放因子
    try:
        # 设置 DPO adapter 的缩放因子为 0.5（降低 DPO 的影响）
        model.set_adapter("dpo")
        for name, module in model.named_modules():
            if hasattr(module, "scaling") and "dpo" in name:
                for k, v in module.scaling.items():
                    module.scaling[k] = 0.5

        model.set_adapter(["sft", "dpo"])
        response_weighted = test_dual_adapter(model, tokenizer, args.instruction, ["sft", "dpo"])
        print(f"(DPO scaled to 0.5)\n{response_weighted}")
    except Exception as e:
        print(f"Weighted test failed: {e}")

    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
