#!/usr/bin/env python3
"""
下载并采样 Benchmark 评测数据
从主流 benchmark 中随机采样 1/10 的数据
"""
import json
import random
import os
import time
from pathlib import Path

from src.logger import setup_logger

# 初始化日志
logger = None

# 尝试导入 datasets
try:
    from datasets import load_dataset
    print("✓ datasets 库可用")
except ImportError:
    print("✗ 请安装：pip install datasets")
    exit(1)


# 定义要下载的 benchmark 数据集
BENCHMARKS = {
    # 中文能力
    "ceval": {
        "name": "ceval-valid",
        "config": "ceval-valid",
        "subset": "middle_politics",  # 先下载一个子集作为示例
        "input_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
    },
    "cmmlu": {
        "name": "cmmlu",
        "config": "default",
        "subset": "agronomy",  # 先下载一个子集
        "input_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
    },
    # 推理能力
    "gsm8k": {
        "name": "gsm8k",
        "config": "main",
        "input_field": "question",
        "choices_field": None,
        "answer_field": "answer",
    },
    # 常识推理
    "hellaswag": {
        "name": "hellaswag",
        "config": "default",
        "input_field": "ctx",
        "choices_field": "endings",
        "answer_field": "label",
    },
    # 科学问答
    "arc_e": {
        "name": "ai2_arc",
        "config": "ARC-Easy",
        "input_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
    },
    "arc_c": {
        "name": "ai2_arc",
        "config": "ARC-Challenge",
        "input_field": "question",
        "choices_field": "choices",
        "answer_field": "answerKey",
    },
    # 事实性
    "truthfulqa": {
        "name": "truthful_qa",
        "config": "multiple_choice",
        "input_field": "question",
        "choices_field": "mc_targets",
        "answer_field": "mc_targets",
    },
    # 综合知识 (MMLU 太大，只采样少量)
    "mmlu": {
        "name": "cais/mmlu",
        "config": "high_school_geography",
        "input_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
    },
}


def sample_dataset(data, sample_ratio=0.1, max_samples=None):
    """随机采样数据集"""
    if isinstance(data, list):
        n = len(data)
        if max_samples:
            n = min(n, max_samples)
        else:
            n = max(1, int(n * sample_ratio))
        return random.sample(data, n)
    return data


def format_multiple_choice_question(question, choices, answer=None):
    """格式化选择题"""
    if isinstance(choices, dict):
        # TruthfulQA 格式
        if "choices" in choices:
            choices = choices["choices"]
        if "label" in choices:
            answer = choices["label"]

    if isinstance(choices, list):
        options = ""
        for i, choice in enumerate(choices):
            if isinstance(choice, dict):
                choice = choice.get("text", str(choice))
            options += f"  {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}. {choice}\n"

        prompt = f"{question}\n\nOptions:\n{options}"
        return prompt

    return question


def download_and_sample(name, config, sample_ratio=0.1, max_samples=200):
    """下载并采样单个数据集"""
    print(f"\n{'='*50}")
    print(f"Downloading: {name}/{config}")
    print(f"{'='*50}")

    try:
        # 加载数据集
        if "/" in name:
            # 带命名空间的数据集
            dataset = load_dataset(name, config, split="validation", trust_remote_code=True)
        else:
            dataset = load_dataset(name, config, split="validation", trust_remote_code=True)

        print(f"Original size: {len(dataset)} samples")

        # 采样
        if max_samples:
            n = min(len(dataset), max_samples)
        else:
            n = max(1, int(len(dataset) * sample_ratio))

        dataset = dataset.shuffle(seed=42).select(range(n))
        print(f"Sampled size: {n} samples ({sample_ratio*100:.0f}%)")

        return dataset

    except Exception as e:
        print(f"Error loading {name}/{config}: {e}")
        return None


def process_ceval(dataset, n_samples=200):
    """处理 C-Eval 数据集"""
    results = []
    categories = set()

    # 获取所有子集
    for item in dataset:
        categories.add(item.get("subject", "unknown"))

    categories = list(categories)
    print(f"Found categories: {categories[:10]}...")  # 显示前 10 个

    # 每个类别采样
    samples_per_cat = max(5, n_samples // len(categories))

    for cat in categories:
        cat_data = [x for x in dataset if x.get("subject") == cat]
        if not cat_data:
            continue

        sampled = random.sample(cat_data, min(samples_per_cat, len(cat_data)))

        for item in sampled:
            question = item.get("question", "")
            choices = [
                item.get("A", ""),
                item.get("B", ""),
                item.get("C", ""),
                item.get("D", ""),
            ]
            answer = item.get("answer", "")

            # 格式化问题
            options = "\n".join([
                f"  A. {choices[0]}",
                f"  B. {choices[1]}",
                f"  C. {choices[2]}",
                f"  D. {choices[3]}",
            ])

            prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with A, B, C, or D."

            results.append({
                "benchmark": "ceval",
                "category": cat,
                "type": "multiple_choice",
                "prompt": prompt,
                "answer": answer,
                "choices": choices,
            })

    return results


def process_cmmlu(dataset, n_samples=200):
    """处理 CMMLU 数据集"""
    results = []
    categories = set()

    for item in dataset:
        categories.add(item.get("category", "unknown"))

    categories = list(categories)
    print(f"Found categories: {categories[:10]}...")

    samples_per_cat = max(5, n_samples // len(categories))

    for cat in categories:
        cat_data = [x for x in dataset if x.get("category") == cat]
        if not cat_data:
            continue

        sampled = random.sample(cat_data, min(samples_per_cat, len(cat_data)))

        for item in sampled:
            question = item.get("question", "")
            choices = [
                item.get("A", ""),
                item.get("B", ""),
                item.get("C", ""),
                item.get("D", ""),
            ]
            answer = item.get("answer", "")

            options = "\n".join([
                f"  A. {choices[0]}",
                f"  B. {choices[1]}",
                f"  C. {choices[2]}",
                f"  D. {choices[3]}",
            ])

            prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with A, B, C, or D."

            results.append({
                "benchmark": "cmmlu",
                "category": cat,
                "type": "multiple_choice",
                "prompt": prompt,
                "answer": answer,
                "choices": choices,
            })

    return results


def process_gsm8k(dataset, n_samples=200):
    """处理 GSM8K 数学数据集"""
    results = []
    sampled = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    for item in sampled:
        question = item.get("question", "")
        answer = item.get("answer", "")

        # 提取数值答案
        import re
        match = re.search(r'####\s*(\d+)', answer)
        numeric_answer = match.group(1) if match else answer

        results.append({
            "benchmark": "gsm8k",
            "category": "math",
            "type": "math_word_problem",
            "prompt": f"{question}\n\nPlease solve step by step. Put your final answer as a single number at the end.",
            "answer": numeric_answer,
            "full_answer": answer,
        })

    return results


def process_hellaswag(dataset, n_samples=200):
    """处理 Hellaswag 常识推理数据集"""
    results = []
    sampled = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    for item in sampled:
        ctx = item.get("ctx", "")
        endings = item.get("endings", [])
        label = item.get("label", "")

        # 清理文本
        ctx = ctx.strip()
        if not ctx.endswith("."):
            ctx += "."

        prompt = f"Complete the following text:\n\n{ctx}\n\nOptions:\n"
        for i, ending in enumerate(endings):
            prompt += f"  {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}. {ending}\n"
        prompt += "\nChoose the most logical completion. Answer with A, B, C, or D."

        results.append({
            "benchmark": "hellaswag",
            "category": "commonsense",
            "type": "text_completion",
            "prompt": prompt,
            "answer": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[int(label)] if label.isdigit() else label,
            "choices": endings,
        })

    return results


def process_arc(dataset, n_samples=200, benchmark_name="arc"):
    """处理 ARC 科学问答数据集"""
    results = []

    for item in dataset:
        question = item.get("question", "")
        choices = item.get("choices", {})
        answer = item.get("answerKey", "")

        if isinstance(choices, dict):
            choice_texts = choices.get("text", [])
        else:
            choice_texts = choices

        options = "\n".join([
            f"  {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}. {choice_texts[i]}"
            for i in range(min(4, len(choice_texts)))
        ])

        prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with A, B, C, or D."

        results.append({
            "benchmark": benchmark_name,
            "category": "science",
            "type": "multiple_choice",
            "prompt": prompt,
            "answer": answer,
            "choices": choice_texts[:4],
        })

    return results


def process_truthfulqa(dataset, n_samples=100):
    """处理 TruthfulQA 事实性数据集"""
    results = []
    sampled = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))

    for item in sampled:
        question = item.get("question", "")
        # TruthfulQA 使用 mc1_targets (单选择) 和 mc2_targets (多选择)
        mc1_targets = item.get("mc1_targets", {})

        choices = mc1_targets.get("choices", [])
        labels = mc1_targets.get("labels", [])

        if not choices or len(choices) == 0:
            continue

        # 找到正确答案的索引
        correct_idx = 0
        for i, label in enumerate(labels):
            if label == 1:
                correct_idx = i
                break

        # 限制到 4 个选项
        choices = list(choices)[:4]
        if len(choices) < 2:
            continue

        options = "\n".join([
            f"  {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}. {choice}"
            for i, choice in enumerate(choices)
        ])

        prompt = f"{question}\n\nOptions:\n{options}\n\nChoose the most factually correct answer. Answer with A, B, C, or D."

        results.append({
            "benchmark": "truthfulqa",
            "category": "factual",
            "type": "multiple_choice",
            "prompt": prompt,
            "answer": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[correct_idx],
            "choices": choices,
        })

    return results


def process_mmlu(dataset, n_samples=200):
    """处理 MMLU 综合知识数据集"""
    results = []

    for item in dataset:
        question = item.get("question", "")
        choices = item.get("choices", [])
        answer = item.get("answer", "")

        if not choices or len(choices) < 2:
            continue

        options = "\n".join([
            f"  {'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}. {choice}"
            for i, choice in enumerate(choices[:4])
        ])

        prompt = f"{question}\n\nOptions:\n{options}\n\nAnswer with A, B, C, or D."

        results.append({
            "benchmark": "mmlu",
            "category": "general_knowledge",
            "type": "multiple_choice",
            "prompt": prompt,
            "answer": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[answer] if isinstance(answer, int) else answer,
            "choices": choices[:4],
        })

    return results


def main():
    global logger
    random.seed(42)

    # 创建输出目录和日志
    output_dir = Path("./bench_test_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir, name="download_benchmark")

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Benchmark Test Data Downloader")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir.absolute()}")
    logger.info("\nDownloading and sampling ~10% of each benchmark...")

    all_data = []
    summary = {}

    # 1. C-Eval (中文) - 使用 OpenDataLab 镜像
    logger.info("\n" + "=" * 50)
    logger.info("1. C-Eval (中文综合评测)")
    logger.info("=" * 50)
    try:
        # 尝试多个镜像源
        for repo in ["ceval/ceval-valid", "muhanwu/CEval", "davidcanedo/ceval-valid"]:
            try:
                ceval_dataset = load_dataset(repo, split="validation")
                logger.info(f"Loaded from {repo}: {len(ceval_dataset)} samples")
                break
            except Exception:
                continue
        else:
            raise Exception("All mirrors failed")

        ceval_data = process_ceval(ceval_dataset, n_samples=200)
        logger.info(f"Sampled {len(ceval_data)} samples")
        all_data.extend(ceval_data)
        summary["ceval"] = len(ceval_data)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.warning("Skipping C-Eval, will use alternative Chinese dataset")
        # 使用 CMMLU 作为替代
        summary["ceval"] = 0

    # 2. CMMLU (中文) - 使用 OpenDataLab 镜像
    logger.info("\n" + "=" * 50)
    logger.info("2. CMMLU (中文多选)")
    logger.info("=" * 50)
    try:
        for repo in ["haonan-li/cmmlu", "MuhanHan/CMMLU"]:
            try:
                cmmlu_dataset = load_dataset(repo, split="validation")
                logger.info(f"Loaded from {repo}: {len(cmmlu_dataset)} samples")
                break
            except Exception:
                continue
        else:
            raise Exception("All mirrors failed")

        cmmlu_data = process_cmmlu(cmmlu_dataset, n_samples=200)
        logger.info(f"Sampled {len(cmmlu_data)} samples")
        all_data.extend(cmmlu_data)
        summary["cmmlu"] = len(cmmlu_data)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.warning("Skipping CMMLU")
        summary["cmmlu"] = 0

    # 3. GSM8K (数学推理)
    logger.info("\n" + "=" * 50)
    logger.info("3. GSM8K (小学数学推理)")
    logger.info("=" * 50)
    try:
        gsm8k_dataset = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
        logger.info(f"Loaded {len(gsm8k_dataset)} samples")
        gsm8k_data = process_gsm8k(gsm8k_dataset, n_samples=200)
        logger.info(f"Sampled {len(gsm8k_data)} samples")
        all_data.extend(gsm8k_data)
        summary["gsm8k"] = len(gsm8k_data)
    except Exception as e:
        logger.error(f"Error: {e}")

    # 4. Hellaswag (常识推理)
    logger.info("\n" + "=" * 50)
    logger.info("4. Hellaswag (常识推理)")
    logger.info("=" * 50)
    try:
        hellaswag_dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
        logger.info(f"Loaded {len(hellaswag_dataset)} samples")
        hellaswag_data = process_hellaswag(hellaswag_dataset, n_samples=200)
        logger.info(f"Sampled {len(hellaswag_data)} samples")
        all_data.extend(hellaswag_data)
        summary["hellaswag"] = len(hellaswag_data)
    except Exception as e:
        logger.error(f"Error: {e}")

    # 5. ARC-Easy (科学问答)
    logger.info("\n" + "=" * 50)
    logger.info("5. ARC-Easy (科学问答)")
    logger.info("=" * 50)
    try:
        arc_e_dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation", trust_remote_code=True)
        logger.info(f"Loaded {len(arc_e_dataset)} samples")
        arc_e_data = process_arc(arc_e_dataset, n_samples=200, benchmark_name="arc_easy")
        logger.info(f"Sampled {len(arc_e_data)} samples")
        all_data.extend(arc_e_data)
        summary["arc_easy"] = len(arc_e_data)
    except Exception as e:
        logger.error(f"Error: {e}")

    # 6. ARC-Challenge (科学问答)
    logger.info("\n" + "=" * 50)
    logger.info("6. ARC-Challenge (科学问答)")
    logger.info("=" * 50)
    try:
        arc_c_dataset = load_dataset("ai2_arc", "ARC-Challenge", split="validation", trust_remote_code=True)
        logger.info(f"Loaded {len(arc_c_dataset)} samples")
        arc_c_data = process_arc(arc_c_dataset, n_samples=200, benchmark_name="arc_challenge")
        logger.info(f"Sampled {len(arc_c_data)} samples")
        all_data.extend(arc_c_data)
        summary["arc_challenge"] = len(arc_c_data)
    except Exception as e:
        logger.error(f"Error: {e}")

    # 7. TruthfulQA (事实性)
    logger.info("\n" + "=" * 50)
    logger.info("7. TruthfulQA (事实性/抗幻觉)")
    logger.info("=" * 50)
    try:
        truthfulqa_dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        logger.info(f"Loaded {len(truthfulqa_dataset)} samples")
        truthfulqa_data = process_truthfulqa(truthfulqa_dataset, n_samples=100)
        logger.info(f"Sampled {len(truthfulqa_data)} samples")
        all_data.extend(truthfulqa_data)
        summary["truthfulqa"] = len(truthfulqa_data)
    except Exception as e:
        logger.error(f"Error: {e}")
        summary["truthfulqa"] = 0

    # 8. MMLU (综合知识)
    logger.info("\n" + "=" * 50)
    logger.info("8. MMLU (57 学科综合知识)")
    logger.info("=" * 50)
    try:
        # MMLU 太大，只下载一个子集
        mmlu_dataset = load_dataset("cais/mmlu", "high_school_geography", split="test", trust_remote_code=True)
        logger.info(f"Loaded {len(mmlu_dataset)} samples (high_school_geography only)")
        mmlu_data = process_mmlu(mmlu_dataset, n_samples=100)
        logger.info(f"Sampled {len(mmlu_data)} samples")
        all_data.extend(mmlu_data)
        summary["mmlu"] = len(mmlu_data)
    except Exception as e:
        logger.error(f"Error: {e}")

    # 保存数据
    logger.info("\n" + "=" * 60)
    logger.info("Saving data...")
    logger.info("=" * 60)

    # 保存合并的数据
    combined_path = output_dir / "combined_benchmarks.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"\n✓ Combined data: {combined_path} ({len(all_data)} samples)")

    # 保存每个 benchmark 的单独文件
    benchmark_data = {}
    for item in all_data:
        benchmark = item["benchmark"]
        if benchmark not in benchmark_data:
            benchmark_data[benchmark] = []
        benchmark_data[benchmark].append(item)

    for benchmark, data in benchmark_data.items():
        path = output_dir / f"{benchmark}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"✓ {benchmark}: {path} ({len(data)} samples)")

    # 保存摘要
    summary["total"] = len(all_data)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✓ Summary: {summary_path}")

    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for benchmark, count in summary.items():
        if benchmark != "total":
            logger.info(f"  {benchmark}: {count} samples")
    logger.info(f"  TOTAL: {summary['total']} samples")
    logger.info("=" * 60)

    # 显示耗时
    elapsed = time.time() - start_time
    logger.info(f"\nDone! Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info("Done!")


if __name__ == "__main__":
    main()
