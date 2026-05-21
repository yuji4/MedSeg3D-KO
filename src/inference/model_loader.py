from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


DEFAULT_MODEL = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
PROJ_OUT_NUM = 256  # <im_patch> 토큰 수 (모델 고정값)


@dataclass
class ModelConfig:
    model_name_or_path: str = DEFAULT_MODEL
    precision: str = "bf16"          # "fp32" | "fp16" | "bf16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    device: str = "cuda"
    max_length: int = 512
    proj_out_num: int = PROJ_OUT_NUM
    cache_dir: str | None = None     # HF 캐시 디렉토리 (Colab: "/content/drive/…")


def _build_dtype(config: ModelConfig) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(
        config.precision, torch.bfloat16
    )


def _build_quant_kwargs(config: ModelConfig, dtype: torch.dtype) -> dict:
    if config.load_in_4bit:
        return {
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    if config.load_in_8bit:
        return {
            "torch_dtype": torch.float16,
            "quantization_config": BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    return {"torch_dtype": dtype}


def load_model(config: ModelConfig | None = None) -> tuple:
    """
    M3D-LaMed 모델과 토크나이저를 로드하여 반환.

    Returns:
        (model, tokenizer) — model은 eval 모드, device는 config.device
    """
    if config is None:
        config = ModelConfig()

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    dtype = _build_dtype(config)
    quant_kwargs = _build_quant_kwargs(config, dtype)

    hf_kwargs = dict(
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **quant_kwargs,
    )
    if config.cache_dir:
        hf_kwargs["cache_dir"] = config.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        model_max_length=config.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **hf_kwargs,
    )
    model = model.to(device=device)
    model.eval()

    return model, tokenizer


def get_colab_config(cache_dir: str = "/content/drive/MyDrive/M3D_cache") -> ModelConfig:
    """Google Colab T4 GPU 권장 설정 (4-bit 양자화)."""
    return ModelConfig(
        precision="fp16",
        load_in_4bit=True,
        device="cuda",
        cache_dir=cache_dir,
    )
