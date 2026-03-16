"""
推理模块
"""
import torch
import sys
from typing import Optional, List, Generator
from pathlib import Path
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class SFTInference:
    """
    SFT 模型推理类

    Example:
        inference = SFTInference(
            base_model_path="./models/qwen2.5-0.5b-instruct",
            adapter_path="./outputs",
        )
        response = inference.generate("Explain machine learning")
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        """
        初始化推理类

        Args:
            base_model_path: 基座模型路径
            adapter_path: LoRA Adapter 路径 (可选)
            torch_dtype: 数据类型
            device_map: 设备映射
        """
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left",
        )

        # 加载基座模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
        )

        # 加载 Adapter (如果有)
        if adapter_path and Path(adapter_path).exists():
            self.model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
            )
            print(f"Loaded adapter from: {adapter_path}")
        else:
            self.model = self.base_model
            print("Using base model without adapter")

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
    ) -> str:
        """
        生成响应

        Args:
            instruction: 指令
            input_text: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: Top-p 参数
            top_k: Top-k 参数
            do_sample: 是否采样
            repetition_penalty: 重复惩罚

        Returns:
            生成的响应文本
        """
        prompt = self._format_prompt(instruction, input_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,  # 启用 KV Cache - 关键优化！
            num_beams=1,     # 禁用束搜索，加快生成
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # 跳过输入部分，只解码生成的内容
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response

    @torch.no_grad()
    def stream_generate(
        self,
        instruction: str,
        input_text: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, None]:
        """
        流式生成响应 - 实时输出生成的 token，提升感知速度

        Args:
            instruction: 指令
            input_text: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度
            top_p: Top-p 参数
            top_k: Top-k 参数
            repetition_penalty: 重复惩罚

        Yields:
            逐字生成的响应文本
        """
        prompt = self._format_prompt(instruction, input_text)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        # 使用 KV Cache 手动实现流式生成
        input_length = inputs["input_ids"].shape[1]
        curr_ids = inputs["input_ids"].clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            outputs = self.model(
                curr_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # 应用温度
            if temperature > 0:
                next_token_logits /= temperature

            # Top-k + Top-p 采样
            if temperature > 0:
                probs = torch.softmax(next_token_logits, dim=-1)

                # Top-p 采样
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = -float("inf")

                # Top-k 采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 检查是否生成 EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            curr_ids = next_token
            past_key_values = outputs.past_key_values

            # 解码并 yield 当前 token
            yield self.tokenizer.decode(next_token[0], skip_special_tokens=True)


    def batch_generate(
        self,
        instructions: List[str],
        input_texts: Optional[List[str]] = None,
        **generate_kwargs,
    ) -> List[str]:
        """
        批量生成

        Args:
            instructions: 指令列表
            input_texts: 输入文本列表 (可选)
            **generate_kwargs: 生成参数

        Returns:
            响应列表
        """
        if input_texts is None:
            input_texts = [""] * len(instructions)

        responses = []
        for instruction, input_text in zip(instructions, input_texts):
            response = self.generate(instruction, input_text, **generate_kwargs)
            responses.append(response)

        return responses

    def _format_prompt(
        self,
        instruction: str,
        input_text: str = "",
    ) -> str:
        """格式化提示"""
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Output:\n"

    def chat(
        self,
        instruction: str,
        input_text: str = "",
        **generate_kwargs,
    ) -> str:
        """
        对话接口

        Args:
            instruction: 指令
            input_text: 输入文本
            **generate_kwargs: 生成参数

        Returns:
            模型响应
        """
        response = self.generate(
            instruction=instruction,
            input_text=input_text,
            **generate_kwargs,
        )
        return response

    def interactive_chat(self) -> None:
        """交互式对话模式（流式输出）"""
        print("=" * 60)
        print("SFT Model Interactive Chat (Streaming)")
        print("Type 'quit' or 'exit' to exit")
        print("=" * 60)

        while True:
            try:
                instruction = input("\n[Instruction]: ").strip()

                if instruction.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break

                input_text = input("[Input] (optional, press Enter to skip): ").strip()

                print("\n[Response]: ", end="", flush=True)

                # 使用流式生成
                for token in self.stream_generate(
                    instruction=instruction,
                    input_text=input_text if input_text else "",
                    max_new_tokens=512,  # 增加生成长度
                ):
                    print(token, end="", flush=True)
                print()  # 换行

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[Error]: {e}")


def load_model_for_inference(
    model_path: str,
    adapter_path: Optional[str] = None,
    device: str = "cuda",
) -> SFTInference:
    """
    加载模型用于推理

    Args:
        model_path: 模型路径
        adapter_path: Adapter 路径
        device: 设备

    Returns:
        SFTInference 对象
    """
    return SFTInference(
        base_model_path=model_path,
        adapter_path=adapter_path,
    )
