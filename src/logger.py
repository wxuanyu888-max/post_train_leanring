"""
训练日志模块
提供结构化日志、TensorBoard 支持、自定义回调和训练报告生成
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict

from transformers import TrainerCallback, TrainerState, TrainingArguments


# ============================================================================
# 日志配置
# ============================================================================

def setup_logger(output_dir: str, name: str = "sft_training") -> logging.Logger:
    """
    配置训练日志记录器

    Args:
        output_dir: 日志输出目录
        name: 日志记录器名称

    Returns:
        配置好的 Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 控制台处理器 - 彩色输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件处理器 - 普通文本
    log_file = Path(output_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


class ColoredFormatter(logging.Formatter):
    """彩色日志格式器"""

    # ANSI 颜色代码
    COLORS = {
        "DEBUG": "",
        "INFO": "\033[36m",      # 青色
        "WARNING": "\033[33m",   # 黄色
        "ERROR": "\033[31m",     # 红色
        "CRITICAL": "\033[35m",  # 紫色
        "RESET": "\033[0m",      # 重置
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


# ============================================================================
# 结构化日志回调
# ============================================================================

@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    step: int = 0
    epoch: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    tokens_per_second: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class StructuredLoggingCallback(TrainerCallback):
    """
    结构化日志回调
    将训练日志保存为 JSON 格式，便于后续分析和可视化
    """

    def __init__(self, output_dir: str, logger: Optional[logging.Logger] = None):
        """
        Args:
            output_dir: 输出目录
            logger: 日志记录器
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("sft_training")
        self.metrics_log: List[Dict] = []
        self.start_time = None
        self.start_step = 0
        self.start_tokens = 0

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """训练开始"""
        self.start_time = time.time()
        self.start_step = state.global_step
        self.start_tokens = state.total_flos

        self.logger.info("=" * 60)
        self.logger.info("训练开始")
        self.logger.info("=" * 60)
        self.logger.info(f"总步数：{state.max_steps}")
        self.logger.info(f"训练轮次：{state.num_train_epochs}")
        self.logger.info(f"初始步数：{state.global_step}")

        # 保存训练开始信息
        self._save_train_info(args, state)

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        """日志记录"""
        if logs is None:
            return

        # 提取指标
        metrics = self._extract_metrics(state, logs)
        self.metrics_log.append(asdict(metrics))

        # 格式化输出
        self._print_log_line(metrics)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """Epoch 结束"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        steps_done = state.global_step - self.start_step
        steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

        self.logger.info("-" * 40)
        self.logger.info(f"Epoch {state.epoch:.2f} 完成")
        self.logger.info(f"已用时间：{self._format_time(elapsed)}")
        self.logger.info(f"平均速度：{steps_per_sec:.2f} steps/sec")
        self.logger.info("-" * 40)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control, metrics, **kwargs):
        """评估完成"""
        eval_loss = metrics.get("eval_loss")
        eval_ppl = metrics.get("eval_perplexity")

        self.logger.info(f"评估结果:")
        self.logger.info(f"  eval_loss = {eval_loss:.4f}" if eval_loss else "")
        self.logger.info(f"  eval_ppl  = {eval_ppl:.4f}" if eval_ppl else "")

        # 更新最近一条指标的评估数据
        if self.metrics_log and eval_loss is not None:
            self.metrics_log[-1]["eval_loss"] = eval_loss
            if eval_ppl is not None:
                self.metrics_log[-1]["eval_perplexity"] = eval_ppl

    def on_save(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """保存 checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{state.global_step}"
        self.logger.info(f"模型已保存至：checkpoint-{state.global_step}")

        # 保存当前的 metrics log
        self._save_metrics_log()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """训练结束"""
        elapsed = time.time() - self.start_time if self.start_time else 0

        self.logger.info("=" * 60)
        self.logger.info("训练完成")
        self.logger.info("=" * 60)
        self.logger.info(f"总用时：{self._format_time(elapsed)}")
        self.logger.info(f"最终步数：{state.global_step}")

        # 保存最终日志
        self._save_metrics_log()
        self._save_training_summary(args, state, elapsed)

    def _extract_metrics(self, state: TrainerState, logs: Dict) -> TrainingMetrics:
        """从日志中提取指标"""
        metrics = TrainingMetrics(
            step=state.global_step,
            epoch=state.epoch,
            loss=logs.get("loss", 0.0),
            learning_rate=logs.get("learning_rate", 0.0),
            grad_norm=logs.get("grad_norm"),
        )

        # GPU 显存
        if torch := __import__("torch"):
            if torch.cuda.is_available():
                metrics.gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2

        return metrics

    def _print_log_line(self, metrics: TrainingMetrics):
        """打印格式化的日志行"""
        step_str = f"[{metrics.step:5d}]"
        epoch_str = f"ep={metrics.epoch:.2f}"
        loss_str = f"loss={metrics.loss:.4f}"
        lr_str = f"lr={metrics.learning_rate:.2e}"

        parts = [step_str, epoch_str, loss_str, lr_str]

        if metrics.eval_loss is not None:
            parts.append(f"eval_loss={metrics.eval_loss:.4f}")
        if metrics.gpu_memory_mb is not None:
            parts.append(f"gpu={metrics.gpu_memory_mb:.0f}MB")

        self.logger.info(" | ".join(parts))

    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}小时{mins}分钟"

    def _save_train_info(self, args: TrainingArguments, state: TrainerState):
        """保存训练开始信息"""
        info = {
            "start_time": datetime.now().isoformat(),
            "output_dir": str(self.output_dir),
            "num_train_epochs": args.num_train_epochs,
            "total_steps": state.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
        }
        with open(self.output_dir / "train_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def _save_metrics_log(self):
        """保存指标日志"""
        with open(self.output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics_log, f, indent=2, ensure_ascii=False)

    def _save_training_summary(self, args: TrainingArguments, state: TrainerState, elapsed: float):
        """保存训练摘要"""
        steps_done = state.global_step - self.start_step

        summary = {
            "end_time": datetime.now().isoformat(),
            "total_time_seconds": elapsed,
            "total_steps": steps_done,
            "steps_per_second": steps_done / elapsed if elapsed > 0 else 0,
            "final_loss": self.metrics_log[-1]["loss"] if self.metrics_log else None,
            "best_eval_loss": min((m.get("eval_loss") for m in self.metrics_log if m.get("eval_loss")), default=None),
        }

        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


# ============================================================================
# TensorBoard 支持
# ============================================================================

class TensorBoardCallback(TrainerCallback):
    """
    TensorBoard 回调
    将训练指标写入 TensorBoard 日志
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Args:
            log_dir: TensorBoard 日志目录
        """
        self.log_dir = log_dir
        self.writer = None
        self._setup_writer()

    def _setup_writer(self):
        """设置 TensorBoard Writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_dir = self.log_dir or "runs/sft_training"
            self.writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            print("警告：无法导入 TensorBoard，请安装：pip install tensorboard")

    def on_log(self, args: TrainingArguments, state: TrainerState, control, logs=None, **kwargs):
        """记录日志到 TensorBoard"""
        if self.writer is None or logs is None:
            return

        # 记录主要指标
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, state.global_step)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control, metrics, **kwargs):
        """记录评估指标到 TensorBoard"""
        if self.writer is None:
            return

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key.startswith("eval_"):
                self.writer.add_scalar(f"eval/{key}", value, state.global_step)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control, **kwargs):
        """训练结束，关闭 Writer"""
        if self.writer:
            self.writer.close()


# ============================================================================
# 训练报告生成
# ============================================================================

def generate_training_report(output_dir: str) -> str:
    """
    生成训练报告

    Args:
        output_dir: 训练输出目录

    Returns:
        训练报告文本
    """
    output_path = Path(output_dir)

    # 加载训练指标
    metrics_file = output_path / "training_metrics.json"
    if not metrics_file.exists():
        return "未找到训练指标文件"

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 加载训练摘要
    summary_file = output_path / "training_summary.json"
    summary = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # 生成报告
    report = []
    report.append("=" * 60)
    report.append("SFT 训练报告")
    report.append("=" * 60)
    report.append("")

    # 基本信息
    report.append("【训练概览】")
    report.append(f"  训练步数：{len(metrics)}")
    if summary:
        report.append(f"  总用时：{summary.get('total_time_seconds', 0):.1f} 秒")
        report.append(f"  平均速度：{summary.get('steps_per_second', 0):.2f} steps/sec")
    report.append("")

    # Loss 趋势
    report.append("【Loss 趋势】")
    if metrics:
        losses = [m.get("loss", 0) for m in metrics]
        report.append(f"  初始 Loss: {losses[0]:.4f}")
        report.append(f"  最终 Loss: {losses[-1]:.4f}")
        report.append(f"  最小 Loss: {min(losses):.4f}")
        report.append(f"  平均 Loss: {sum(losses)/len(losses):.4f}")

        # 计算下降幅度
        if losses[0] > 0:
            drop = (losses[0] - losses[-1]) / losses[0] * 100
            report.append(f"  下降幅度：{drop:.1f}%")
    report.append("")

    # 评估结果
    report.append("【评估结果】")
    eval_losses = [m.get("eval_loss") for m in metrics if m.get("eval_loss")]
    if eval_losses:
        report.append(f"  评估次数：{len(eval_losses)}")
        report.append(f"  最佳 eval_loss: {min(eval_losses):.4f}")
        report.append(f"  最终 eval_loss: {eval_losses[-1]:.4f}")
    else:
        report.append("  无评估数据")
    report.append("")

    # 学习率变化
    report.append("【学习率变化】")
    if metrics:
        lrs = [m.get("learning_rate", 0) for m in metrics]
        report.append(f"  初始 LR: {lrs[0]:.2e}")
        report.append(f"  最终 LR: {lrs[-1]:.2e}")
    report.append("")

    report.append("=" * 60)

    report_text = "\n".join(report)

    # 保存到文件
    report_file = output_path / "training_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ============================================================================
# 便捷函数
# ============================================================================

def get_callbacks(output_dir: str, use_tensorboard: bool = True) -> List[TrainerCallback]:
    """
    获取所有回调函数

    Args:
        output_dir: 输出目录
        use_tensorboard: 是否启用 TensorBoard

    Returns:
        回调函数列表
    """
    logger = setup_logger(output_dir)
    callbacks = [
        StructuredLoggingCallback(output_dir, logger),
    ]

    if use_tensorboard:
        callbacks.append(TensorBoardCallback(os.path.join(output_dir, "tensorboard")))

    return callbacks


# ============================================================================
# Benchmark 评测日志支持
# ============================================================================

@dataclass
class BenchmarkMetrics:
    """评测指标数据类"""
    benchmark_name: str = ""
    total_samples: int = 0
    correct_samples: int = 0
    accuracy: float = 0.0
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    samples_per_second: float = 0.0
    category_breakdown: Dict[str, Dict] = field(default_factory=dict)


class BenchmarkLogger:
    """
    Benchmark 评测日志记录器
    提供统一的评测日志格式和结构化输出
    """

    def __init__(self, output_dir: str, name: str = "benchmark_eval"):
        """
        Args:
            output_dir: 日志输出目录
            name: 日志记录器名称
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(output_dir, name)
        self.results_log: List[Dict] = []
        self.start_time = None
        self.current_benchmark = None

    def start_evaluation(self, benchmark_name: str, total_samples: int):
        """开始评测"""
        self.current_benchmark = benchmark_name
        self.start_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info(f"Benchmark 评测开始：{benchmark_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"总样本数：{total_samples}")
        self.logger.info(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            "benchmark_name": benchmark_name,
            "total_samples": total_samples,
            "start_time": datetime.now().isoformat()
        }

    def log_sample(self, idx: int, prompt: str, expected: str,
                   predicted: str, correct: bool,
                   benchmark_name: str = None, qtype: str = None,
                   response: str = None):
        """
        记录单个样本的评测结果

        Args:
            idx: 样本索引
            prompt: 问题提示
            expected: 期望答案
            predicted: 预测答案（提取后的，如 A/B/C/D）
            correct: 是否正确
            benchmark_name: benchmark 名称
            qtype: 问题类型
            response: LLM 完整原始输出
        """
        if qtype:
            self.logger.info(f"[{idx+1}] 类型：{qtype}")

        # 完整输出问题
        self.logger.info(f"问题：\n{prompt}")
        self.logger.info(f"期望答案：{expected}")
        self.logger.info(f"提取答案：{predicted}")
        self.logger.info(f"评测结果：{'✓ 正确' if correct else '✗ 错误'}")

        # 记录 LLM 完整输出
        if response:
            self.logger.info(f"LLM 完整输出：\n{response}")

        # 记录到日志（JSON 格式保存完整信息）
        result_entry = {
            "idx": idx,
            "benchmark": benchmark_name or self.current_benchmark,
            "type": qtype,
            "prompt": prompt,
            "expected": expected,
            "predicted": predicted,
            "response": response,  # 完整 LLM 输出
            "correct": correct,
            "timestamp": datetime.now().isoformat()
        }
        self.results_log.append(result_entry)

        # 每记录一个样本就立即保存到文件（增量保存）
        self._save_single_result(result_entry)

    def log_progress(self, current: int, total: int, correct: int):
        """记录评测进度"""
        accuracy = correct / current * 100 if current > 0 else 0
        self.logger.info(f">>> 进度：[{current}/{total}] 正确率：{accuracy:.1f}% <<<")

    def end_evaluation(self, total_samples: int, correct_samples: int,
                       category_breakdown: Dict[str, Dict] = None) -> BenchmarkMetrics:
        """
        结束评测并生成摘要

        Args:
            total_samples: 总样本数
            correct_samples: 正确样本数
            category_breakdown: 按类别分类的统计

        Returns:
            BenchmarkMetrics 对象
        """
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0

        accuracy = correct_samples / total_samples * 100 if total_samples > 0 else 0
        samples_per_sec = total_samples / duration if duration > 0 else 0

        metrics = BenchmarkMetrics(
            benchmark_name=self.current_benchmark or "unknown",
            total_samples=total_samples,
            correct_samples=correct_samples,
            accuracy=accuracy,
            start_time=datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else "",
            end_time=datetime.now().isoformat(),
            duration_seconds=duration,
            samples_per_second=samples_per_sec,
            category_breakdown=category_breakdown or {}
        )

        self.logger.info("=" * 60)
        self.logger.info(f"Benchmark 评测完成：{metrics.benchmark_name}")
        self.logger.info("=" * 60)
        self.logger.info(f"总样本数：{metrics.total_samples}")
        self.logger.info(f"正确样本：{metrics.correct_samples}")
        self.logger.info(f"正确率：{metrics.accuracy:.2f}%")
        self.logger.info(f"耗时：{duration:.1f}秒 ({samples_per_sec:.2f} samples/sec)")

        if category_breakdown:
            self.logger.info("\n按类别统计:")
            for cat, stats in category_breakdown.items():
                cat_acc = stats.get('accuracy', 0)
                cat_total = stats.get('total', 0)
                cat_correct = stats.get('correct', 0)
                self.logger.info(f"  {cat}: {cat_acc:.1f}% ({cat_correct}/{cat_total})")

        # 保存评测结果
        self._save_results()
        self._save_summary(metrics)

        return metrics

    def _save_results(self):
        """保存详细评测结果"""
        results_file = self.output_dir / "benchmark_results.json"

        # 如果文件已存在，加载现有数据
        existing_results = []
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                existing_results = json.load(f)

        existing_results.extend(self.results_log)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"评测结果已保存：{results_file}")

    def _save_single_result(self, result_entry: Dict):
        """增量保存单个评测结果"""
        results_file = self.output_dir / "benchmark_results.json"

        # 如果文件已存在，加载现有数据
        existing_results = []
        if results_file.exists():
            try:
                with open(results_file, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                # 文件损坏，从头开始
                existing_results = []

        # 追加新结果
        existing_results.append(result_entry)

        # 立即写回文件
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)

    def _save_summary(self, metrics: BenchmarkMetrics):
        """保存评测摘要"""
        summary_file = self.output_dir / "benchmark_summary.json"

        # 加载现有摘要
        existing_summary = {}
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)

        # 更新当前 benchmark 的摘要
        existing_summary[metrics.benchmark_name] = {
            "total_samples": metrics.total_samples,
            "correct_samples": metrics.correct_samples,
            "accuracy": metrics.accuracy,
            "start_time": metrics.start_time,
            "end_time": metrics.end_time,
            "duration_seconds": metrics.duration_seconds,
            "samples_per_second": metrics.samples_per_second,
            "category_breakdown": metrics.category_breakdown,
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(existing_summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"评测摘要已保存：{summary_file}")


def print_benchmark_summary(summary_file: str):
    """
    打印 benchmark 评测摘要

    Args:
        summary_file: 摘要文件路径
    """
    path = Path(summary_file)
    if not path.exists():
        print(f"错误：未找到摘要文件 {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    print("\n" + "=" * 60)
    print("Benchmark 评测汇总")
    print("=" * 60)

    total_correct = 0
    total_samples = 0

    for benchmark, stats in summary.items():
        acc = stats.get('accuracy', 0)
        correct = stats.get('correct_samples', 0)
        total = stats.get('total_samples', 0)
        duration = stats.get('duration_seconds', 0)

        print(f"\n{benchmark}:")
        print(f"  正确率：{acc:.2f}% ({correct}/{total})")
        print(f"  耗时：{duration:.1f}秒")

        # 类别细分
        if 'category_breakdown' in stats and stats['category_breakdown']:
            print("  按类别:")
            for cat, cat_stats in stats['category_breakdown'].items():
                cat_acc = cat_stats.get('accuracy', 0)
                cat_correct = cat_stats.get('correct', 0)
                cat_total = cat_stats.get('total', 0)
                print(f"    {cat}: {cat_acc:.1f}% ({cat_correct}/{cat_total})")

        total_correct += correct
        total_samples += total

    print("\n" + "=" * 60)
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"总计：{overall_acc:.2f}% ({total_correct}/{total_samples})")
    print("=" * 60)
