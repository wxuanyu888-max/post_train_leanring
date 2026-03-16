#!/usr/bin/env python3
"""
DPO 数据准备脚本
从多种来源生成 DPO 偏好数据集
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Optional

from datasets import load_dataset


def load_jsonl(path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], path: str):
    """保存 JSONL 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ==================== 方法 1: 从 SFT 数据转换 ====================

def convert_sft_to_dpo_with_rejected_samples(
    sft_data_path: str,
    output_path: str,
    model_for_rejection=None,
    num_rejected_per_sample: int = 2,
) -> List[Dict]:
    """
    从 SFT 数据生成 DPO 数据

    策略：
    - chosen: 原始 SFT 数据中的高质量回答
    - rejected: 通过模型生成的低质量回答（添加噪声、截断等）

    Args:
        sft_data_path: SFT 数据路径
        output_path: 输出路径
        model_for_rejection: 用于生成 rejected 的模型（可选）
        num_rejected_per_sample: 每个样本生成多少个 rejected

    Returns:
        DPO 数据列表
    """
    sft_data = load_jsonl(sft_data_path)
    dpo_data = []

    for i, item in enumerate(sft_data):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        chosen = item.get("output", "")

        # 生成多个 rejected 版本
        rejected_list = generate_rejected_responses(
            chosen,
            num_rejected_per_sample,
            model=model_for_rejection,
            prompt=f"{instruction}\n{input_text}" if input_text else instruction
        )

        for rejected in rejected_list:
            dpo_item = {
                "instruction": instruction,
                "input": input_text,
                "chosen": chosen,
                "rejected": rejected,
            }
            dpo_data.append(dpo_item)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(sft_data)} samples")

    save_jsonl(dpo_data, output_path)
    print(f"Saved {len(dpo_data)} DPO samples to {output_path}")
    return dpo_data


def generate_rejected_responses(
    chosen_response: str,
    num_samples: int = 2,
    model=None,
    prompt: str = "",
) -> List[str]:
    """
    生成低质量的 rejected 回复

    策略：
    1. 截断回复
    2. 添加噪声/错误
    3. 生成不完整回复
    4. 使用模型生成但降低质量
    """
    rejected_list = []

    # 策略 1: 截断回复（保留开头）
    if len(chosen_response) > 50:
        truncated = chosen_response[:len(chosen_response) // 2] + "..."
        rejected_list.append(truncated)

    # 策略 2: 移除关键信息
    words = chosen_response.split()
    if len(words) > 10:
        # 随机删除一些词
        masked = chosen_response.replace(
            random.choice(words[5:]),
            "[...]"
        )
        rejected_list.append(masked)

    # 策略 3: 添加语法错误
    noisy = chosen_response.replace("，", " ").replace("。", "，")
    rejected_list.append(noisy[:200] + "..." if len(noisy) > 200 else noisy)

    # 策略 4: 通用低质回复
    generic_rejected = [
        "抱歉，我无法回答这个问题。",
        "这是一个有趣的问题，但我没有足够的信息来回答。",
        "让我想想... 嗯... 我不太确定。",
    ]

    if len(rejected_list) < num_samples:
        for generic in generic_rejected:
            if len(rejected_list) >= num_samples:
                break
            rejected_list.append(generic)

    # 如果提供了模型，用模型生成低质量回复
    if model is not None:
        try:
            # 这里可以调用模型生成
            pass
        except Exception as e:
            print(f"Model generation failed: {e}")

    return rejected_list[:num_samples]


# ==================== 方法 2: 从模型对比输出 ====================

def create_dpo_from_model_comparison(
    prompts_path: str,
    model_a_path: str,
    model_b_path: str,
    output_path: str,
    preference_rule: str = "length",  # length, manual, model_based
) -> List[Dict]:
    """
    从两个模型的输出对比生成 DPO 数据

    Args:
        prompts_path: prompt 数据路径
        model_a_path: 模型 A 输出路径
        model_b_path: 模型 B 输出路径
        output_path: 输出路径
        preference_rule: 偏好规则
            - length: 选择更长的作为 chosen
            - manual: 需要手动标注
            - model_based: 用更强的模型判断

    Returns:
        DPO 数据列表
    """
    prompts = load_jsonl(prompts_path)
    model_a_outputs = load_jsonl(model_a_path)
    model_b_outputs = load_jsonl(model_b_path)

    dpo_data = []

    for i, (prompt_item, output_a, output_b) in enumerate(
        zip(prompts, model_a_outputs, model_b_outputs)
    ):
        instruction = prompt_item.get("instruction", "")
        input_text = prompt_item.get("input", "")

        response_a = output_a.get("output", "")
        response_b = output_b.get("output", "")

        # 根据规则决定偏好
        if preference_rule == "length":
            # 通常更长的回答更详细
            if len(response_a) > len(response_b):
                chosen, rejected = response_a, response_b
            else:
                chosen, rejected = response_b, response_a
        elif preference_rule == "manual":
            # 需要人工标注，这里跳过
            print(f"Sample {i}: Need manual annotation")
            print(f"  A: {response_a[:100]}...")
            print(f"  B: {response_b[:100]}...")
            continue
        else:
            # 默认用长度
            chosen, rejected = (response_a, response_b) if len(response_a) > len(response_b) else (response_b, response_a)

        dpo_item = {
            "instruction": instruction,
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected,
        }
        dpo_data.append(dpo_item)

    save_jsonl(dpo_data, output_path)
    print(f"Saved {len(dpo_data)} DPO samples to {output_path}")
    return dpo_data


# ==================== 方法 3: 使用公开数据集 ====================

def download_and_convert_public_dataset(
    dataset_name: str,
    output_path: str,
    split: str = "train",
) -> List[Dict]:
    """
    下载并转换公开 preference 数据集

    支持的公开数据集:
    - Anthropic/hh-rlhf
    - OpenAssistant/oasst1
    - Stanford/nlp/sft-data
    - llm-blender/ultrafeedback

    Args:
        dataset_name: 数据集名称
        output_path: 输出路径
        split: 数据划分

    Returns:
        DPO 数据列表
    """
    print(f"Downloading {dataset_name}...")

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return []

    dpo_data = []

    # 根据不同数据集格式转换
    if "hh-rlhf" in dataset_name:
        # Anthropic HH-RLHF 格式
        for item in dataset:
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            # HH-RLHF 有 Human 和 Assistant 前缀
            dpo_item = parse_hh_rlhf_format(chosen, rejected)
            if dpo_item:
                dpo_data.append(dpo_item)

    elif "oasst" in dataset_name.lower():
        # OpenAssistant 格式
        dpo_data = convert_oasst_to_dpo(dataset)

    elif "ultrafeedback" in dataset_name.lower():
        # UltraFeedback 格式
        dpo_data = convert_ultrafeedback_to_dpo(dataset)

    else:
        # 通用格式
        for item in dataset:
            if "chosen" in item and "rejected" in item:
                dpo_data.append({
                    "instruction": item.get("prompt", item.get("instruction", "")),
                    "input": "",
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })

    save_jsonl(dpo_data, output_path)
    print(f"Saved {len(dpo_data)} DPO samples to {output_path}")
    return dpo_data


def parse_hh_rlhf_format(chosen: str, rejected: str) -> Optional[Dict]:
    """解析 HH-RLHF 格式"""
    # HH-RLHF 格式：\n\nHuman: xxx\n\nAssistant: yyy
    def extract_assistant_response(text):
        if "Assistant:" in text:
            return text.split("Assistant:")[-1].strip()
        return text.strip()

    chosen_response = extract_assistant_response(chosen)
    rejected_response = extract_assistant_response(rejected)

    # 提取 instruction
    def extract_instruction(text):
        if "Human:" in text:
            return text.split("Human:")[-1].split("Assistant:")[0].strip()
        return text.strip()

    instruction = extract_instruction(chosen)

    return {
        "instruction": instruction,
        "input": "",
        "chosen": chosen_response,
        "rejected": rejected_response,
    }


def convert_oasst_to_dpo(dataset) -> List[Dict]:
    """转换 OpenAssistant 数据"""
    dpo_data = []

    # OASST 数据需要构建成对话树
    # 这里简化处理
    for item in dataset:
        text = item.get("text", "")
        # 简单分割
        parts = text.split("\n")
        if len(parts) >= 2:
            dpo_data.append({
                "instruction": parts[0],
                "input": "",
                "chosen": parts[1] if len(parts) > 1 else "",
                "rejected": parts[-1] if len(parts) > 2 else parts[0],
            })

    return dpo_data


def convert_ultrafeedback_to_dpo(dataset) -> List[Dict]:
    """转换 UltraFeedback 数据"""
    dpo_data = []

    for item in dataset:
        prompt = item.get("prompt", "")
        chosen_list = item.get("chosen", [])
        rejected_list = item.get("rejected", [])

        # UltraFeedback 有多个 chosen/rejected
        if chosen_list and rejected_list:
            chosen = chosen_list[0] if isinstance(chosen_list[0], str) else chosen_list[0].get("response", "")
            rejected = rejected_list[0] if isinstance(rejected_list[0], str) else rejected_list[0].get("response", "")

            dpo_data.append({
                "instruction": prompt,
                "input": "",
                "chosen": chosen,
                "rejected": rejected,
            })

    return dpo_data


# ==================== 方法 4: 人工标注 ====================

def create_annotation_template(output_path: str, sample_size: int = 100):
    """
    创建人工标注模板

    用于人工标注偏好数据
    """
    template = """# DPO 数据人工标注模板

## 标注说明

对于每个样本，请阅读 prompt 和两个模型的回复，然后选择更好的那个作为 chosen。

判断标准：
1. 准确性 - 回答是否正确
2. 完整性 - 是否完整回答了问题
3. 有用性 - 是否对用户有帮助
4. 安全性 - 是否安全无害

## 标注格式

```json
{
  "instruction": "用户指令",
  "input": "输入内容",
  "chosen": "更好的回复",
  "rejected": "较差的回复",
  "reason": "选择理由（可选）"
}
```

## 待标注样本

"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(template)

    print(f"Annotation template saved to {output_path}")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description="Prepare DPO training data")

    parser.add_argument(
        "--method",
        type=str,
        default="sft_convert",
        choices=["sft_convert", "model_compare", "public_dataset", "annotation"],
        help="数据生成方法",
    )

    # SFT 转换参数
    parser.add_argument("--sft_data", type=str, default="./data/sft/train.jsonl")
    parser.add_argument("--output", type=str, default="./data/dpo/train.jsonl")

    # 公开数据集参数
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")

    # 人工标注参数
    parser.add_argument("--annotation_template", type=str, default="./data/dpo/annotation_template.md")

    args = parser.parse_args()

    if args.method == "sft_convert":
        print("Method: Converting SFT data to DPO format...")
        convert_sft_to_dpo_with_rejected_samples(
            sft_data_path=args.sft_data,
            output_path=args.output,
        )

    elif args.method == "public_dataset":
        print(f"Method: Downloading {args.dataset_name}...")
        download_and_convert_public_dataset(
            dataset_name=args.dataset_name,
            output_path=args.output,
        )

    elif args.method == "annotation":
        print("Method: Creating annotation template...")
        create_annotation_template(args.annotation_template)

    elif args.method == "model_compare":
        print("Method: Model comparison (requires model outputs)...")
        print("Please provide model output files for comparison.")


if __name__ == "__main__":
    main()
