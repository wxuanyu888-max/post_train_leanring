"""
模型加载与 LoRA 配置模块
"""
from typing import List, Optional
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)


def load_tokenizer(
    model_path: str,
    padding_side: str = "right",
) -> AutoTokenizer:
    """
    加载分词器

    Args:
        model_path: 模型路径
        padding_side: 填充方向，训练时用 "right"

    Returns:
        分词器对象
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side=padding_side,
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    model_path: str,
    use_4bit: bool = False,
    use_8bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[str] = "auto",
) -> AutoModelForCausalLM:
    """
    加载基座模型

    Args:
        model_path: 模型路径
        use_4bit: 是否使用 4bit 量化 (QLoRA)
        use_8bit: 是否使用 8bit 量化
        torch_dtype: 数据类型
        device_map: 设备映射

    Returns:
        基座模型对象

    Note:
        - use_4bit=True: QLoRA 模式，显存占用最小
        - use_8bit=True: 8bit 量化，显存占用较小
        - 都为 False: 全精度加载，显存占用最大
    """
    # 配置量化参数
    quantization_config = None

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch_dtype,
        )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map if quantization_config else None,
    )

    # 如果使用量化，需要准备模型用于 k-bit 训练
    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    return model


def create_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    创建 LoRA 配置

    Args:
        r: LoRA 秩，控制参数量，越大表达能力越强
        alpha: LoRA 缩放参数，通常为 2*r
        dropout: Dropout 比例，防止过拟合
        target_modules: 目标模块列表
        bias: 是否训练 bias
        task_type: 任务类型

    Returns:
        LoRA 配置对象
    """
    # 默认目标模块 (Qwen2 架构)
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=getattr(TaskType, task_type),
    )


def apply_lora(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig,
) -> AutoModelForCausalLM:
    """
    应用 LoRA 到模型

    Args:
        model: 基座模型
        lora_config: LoRA 配置

    Returns:
        应用 LoRA 后的模型
    """
    return get_peft_model(model, lora_config)


def print_trainable_params(model) -> None:
    """
    打印可训练参数信息

    Args:
        model: 模型对象
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / all_param:.4f}%)")
    print(f"All params: {all_param:,}")
    print(f"Non-trainable params: {all_param - trainable_params:,}")


def get_model_info(model) -> dict:
    """
    获取模型详细信息

    Args:
        model: 模型对象

    Returns:
        包含模型信息的字典
    """
    info = {
        "trainable_params": 0,
        "all_params": 0,
        "layers": {},
    }

    for name, param in model.named_parameters():
        info["all_params"] += param.numel()
        if param.requires_grad:
            info["trainable_params"] += param.numel()

        # 统计每层参数
        layer_name = name.split(".")[0]
        if layer_name not in info["layers"]:
            info["layers"][layer_name] = {"trainable": 0, "all": 0}
        info["layers"][layer_name]["all"] += param.numel()
        if param.requires_grad:
            info["layers"][layer_name]["trainable"] += param.numel()

    return info


def print_lora_info(model) -> None:
    """
    打印 LoRA 模型详细信息

    Args:
        model: LoRA 模型对象
    """
    from peft import get_peft_model_state_dict

    print("\n" + "=" * 50)
    print("LoRA Model Information")
    print("=" * 50)

    # 打印模型配置
    if hasattr(model, "peft_config"):
        config = model.peft_config["default"]
        print(f"LoRA r: {config.r}")
        print(f"LoRA alpha: {config.lora_alpha}")
        print(f"LoRA dropout: {config.lora_dropout}")
        print(f"Target modules: {config.target_modules}")

    # 打印参数统计
    print_trainable_params(model)
    print("=" * 50 + "\n")
