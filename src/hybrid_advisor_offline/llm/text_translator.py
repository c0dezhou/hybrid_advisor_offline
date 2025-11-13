# -*- coding: utf-8 -*-
"""
文案润色工具（受环境变量 USE_LLM_TRANSLATOR 控制）。
提供两种工作模式：
1. 轻量级规则润色：始终可用，完全离线。
2. 本地指令模型润色：默认为 GGUF + llama.cpp 工作流，开启 USE_LLM_TRANSLATOR=1 
   或可切换到Qwen/Qwen2-7B-Instruct（bitsandbytes 4bit） 。，
   若失败会自动回退到规则润色，确保输出稳定。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

TRANSLATOR_ENABLED = os.getenv("USE_LLM_TRANSLATOR", "0") == "1"
BACKEND = os.getenv("LLM_TRANSLATOR_BACKEND", "gguf").strip().lower()

MODEL_ID = os.getenv(
    "LOCAL_TRANSLATOR_MODEL",
    "./models/local_llm/qwen2-7b-instruct-q5_k_m.gguf",
)
MAX_NEW_TOKENS = int(os.getenv("LLM_TRANSLATOR_MAX_NEW_TOKENS", "128"))
TEMPERATURE = float(os.getenv("LLM_TRANSLATOR_TEMPERATURE", "0.4"))

GGUF_GPU_LAYERS = int(os.getenv("GGUF_GPU_LAYERS", "40"))
GGUF_THREADS = int(os.getenv("GGUF_THREADS", "0"))  # 0 -> auto
GGUF_CONTEXT = int(os.getenv("GGUF_CONTEXT_SIZE", "2048"))

_TOKENIZER = None
_MODEL = None
_DEVICE = None


RISK_LABELS = {0: "保守型", 1: "稳健型", 2: "进取型"}


def translator_enabled() -> bool:
    return TRANSLATOR_ENABLED


def _format_alloc(target_alloc) -> str:
    try:
        eq, bd, cs = target_alloc
        return f"股票{int(eq * 100)}%/债券{int(bd * 100)}%/现金{int(cs * 100)}%"
    except Exception:
        return ""


def _rule_based_refine(text: str, context: Dict[str, Any]) -> str:
    """
    使用纯规则的方式做轻量润色，使文案更口语化。
    该方案不依赖任何外部模型，在 LLM 不可用时作为兜底。
    """
    base = text.strip()
    pieces = []

    card_risk = context.get("card_risk_level")
    if card_risk is not None:
        pieces.append(f"{RISK_LABELS.get(card_risk, '稳健型')}卡片")
    if context.get("card_id"):
        pieces.append(str(context["card_id"]))
    header = " · ".join(pieces)

    user_bucket = context.get("user_risk_bucket")
    summary_bits = []
    if header:
        summary_bits.append(header)
    if user_bucket is not None:
        summary_bits.append(f"客户风险等级：{RISK_LABELS.get(user_bucket, '稳健型')}")
    alloc = _format_alloc(context.get("target_alloc"))
    if alloc:
        summary_bits.append(f"配置结构：{alloc}")

    prefix = "；".join(summary_bits)
    refined = base
    if prefix:
        refined = f"{prefix}。{base}"

    refined = refined.replace("建议侧重", "焦点在")
    refined = refined.replace("建议建议", "建议")
    refined = refined.replace("建议保持", "建议保持")
    refined = refined.replace("建议侧重于", "强调")
    return refined


def _ensure_model():
    if BACKEND == "gguf":
        return _ensure_gguf_model()
    return _ensure_hf_model()


def _ensure_hf_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE == "hf":
        return _TOKENIZER, _MODEL, _DEVICE

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"transformers_unavailable: {exc}") from exc

    if MODEL_ID.lower() in ["", "none", "off"]:
        raise RuntimeError("model_id_not_configured")

    print(f"[HF] Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print("[HF] CUDA detected. Using bitsandbytes 4-bit quantization.")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"bitsandbytes_unavailable: {exc}") from exc
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "quantization_config": bnb_config,
        }
        target_device = None
        device_label = "cuda"
    else:
        print("[HF] No CUDA found; loading on CPU (slower).")
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
        }
        target_device = torch.device("cpu")
        device_label = "cpu"

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    except Exception as exc:
        print(f"[HF] Error loading model {MODEL_ID}: {exc}")
        print("[HF] Retrying with CPU float32 fallback ...")
        load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32}
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        target_device = torch.device("cpu")
        device_label = "cpu"

    if target_device is not None:
        model.to(target_device)

    model.eval()
    _TOKENIZER = tokenizer
    _MODEL = model
    _DEVICE = "hf"
    print(f"[HF] Model {MODEL_ID} ready on {device_label}.")
    return _TOKENIZER, _MODEL, device_label


def _ensure_gguf_model():
    global _TOKENIZER, _MODEL, _DEVICE
    if _MODEL is not None and _DEVICE == "gguf":
        return _TOKENIZER, _MODEL, _DEVICE

    try:
        from llama_cpp import Llama
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"llama_cpp_unavailable: {exc}") from exc

    model_path = MODEL_ID
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(
            "gguf_model_not_found: set LOCAL_TRANSLATOR_MODEL to GGUF 文件路径，例如 ./models/local_llm/qwen-7b-instruct-q4_0.gguf"
        )

    print(f"[GGUF] Loading GGUF model from {model_path}")
    model = Llama(
        model_path=model_path,
        n_ctx=GGUF_CONTEXT,
        n_gpu_layers=GGUF_GPU_LAYERS if GGUF_GPU_LAYERS >= 0 else 0,
        n_threads=GGUF_THREADS if GGUF_THREADS > 0 else None,
        chat_format="chatml",
    )

    _TOKENIZER = None
    _MODEL = model
    _DEVICE = "gguf"
    return _TOKENIZER, _MODEL, _DEVICE


def _build_messages(text: str, context: Dict[str, Any]) -> Tuple[str, str]:
    sys_prompt = (
        "你是银行投顾文案润色助手。请在保留合规表述、不加入承诺性词语的前提下，"
        "让解释更自然、强调风险匹配度，让普通人更易读懂。输出仅包含润色后的中文文本。"
        "增加区分度和说服力"
    )
    user_prompt = (
        f"客户风险等级：{RISK_LABELS.get(context.get('user_risk_bucket'), '未知')}\n"
        f"卡片风险等级：{RISK_LABELS.get(context.get('card_risk_level'), '未知')}\n"
        f"目标配置：{_format_alloc(context.get('target_alloc')) or '未提供'}\n"
        f"原始文案：{text.strip()}\n"
        "请在保持含义不变、严格避免夸大承诺的情况下，润色为 1 段中文说明。"
    )
    return sys_prompt, user_prompt


def _build_prompt(tokenizer, text: str, context: Dict[str, Any]):
    sys_prompt, user_prompt = _build_messages(text, context)

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    prompt = sys_prompt + "\n\n" + user_prompt + "\n\n回复："
    return tokenizer(prompt, return_tensors="pt")


def _llm_refine(text: str, context: Dict[str, Any]) -> Tuple[str | None, str]:
    if BACKEND == "gguf":
        return _gguf_refine(text, context)

    try:
        tokenizer, model, device = _ensure_model()
    except Exception as exc:
        return None, f"llm_load_failed:{exc}"

    try:
        inputs = _build_prompt(tokenizer, text, context)
    except Exception as exc:
        return None, f"llm_tokenize_failed:{exc}"

    try:
        import torch

        input_ids = inputs.to(device if device == "cuda" else torch.device("cpu"))
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad_id is None:
            pad_id = getattr(model.config, "eos_token_id", None)
        if isinstance(pad_id, (list, tuple)):
            pad_id = pad_id[0]
        if pad_id is None:
            pad_id = 0
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                pad_token_id=pad_id,
            )
        generated = outputs[0, input_ids.shape[-1]:]
        refined = tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as exc:  # pragma: no cover - GPU/torch errors
        return None, f"llm_generate_failed:{exc}"

    if not refined:
        return None, "llm_empty"
    if refined == text.strip():
        return refined, "llm_nochange"
    return refined, "llm_refined"


def _gguf_refine(text: str, context: Dict[str, Any]) -> Tuple[str | None, str]:
    try:
        _, model, _ = _ensure_gguf_model()
    except Exception as exc:
        return None, f"llm_load_failed:{exc}"

    sys_prompt, user_prompt = _build_messages(text, context)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.9,
        )
        choices = result.get("choices") or []
        if not choices:
            return None, "llm_empty"
        refined = choices[0]["message"]["content"].strip()
    except Exception as exc:  # pragma: no cover
        return None, f"gguf_generate_failed:{exc}"

    if not refined:
        return None, "llm_empty"
    if refined == text.strip():
        return refined, "llm_nochange"
    return refined, "llm_refined"


def refine_text(text: str, context: Dict[str, Any] | None = None) -> Tuple[str, str]:
    """
    返回 (润色后的文本, 状态字符串)。
    状态含义：
      - translator_disabled：未开启 LLM 润色
      - llm_refined / llm_nochange：由 LLM 生成或保持原样
      - rule_refined：由规则润色
      - translator_no_change：规则润色后无变化
      - *_fallback：LLM 异常，已回退至规则模式
    """
    if not text:
        return text, "empty"
    if not TRANSLATOR_ENABLED:
        return text, "translator_disabled"
    context = context or {}

    refined_llm, llm_status = _llm_refine(text, context)
    if llm_status == "llm_refined" and refined_llm:
        return refined_llm, llm_status
    if llm_status == "llm_nochange" and refined_llm:
        return refined_llm, llm_status

    refined_rule = _rule_based_refine(text, context)
    status = "translator_no_change"
    if refined_rule != text:
        status = "rule_refined"
    if llm_status and llm_status not in ["llm_nochange", "llm_refined"]:
        status = f"{llm_status}_fallback"
    return refined_rule, status
