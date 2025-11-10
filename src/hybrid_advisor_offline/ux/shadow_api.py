"""
FastAPI 影子接口：Hybrid Advisor Offline
----------------------------------------

用途：
    - 暴露一个 /recommend 端点，供渠道或审计系统在“影子模式”下查询 CQL 策略输出。
    - 所有响应均附带合规解释文本与审计哈希，方便追踪。

运行方式：
    uvicorn hybrid_advisor_offline.ux.shadow_api:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from hybrid_advisor_offline.llm.text_translator import refine_text
from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import (
    ALL_CARDS,
    get_card_by_id,
)
from hybrid_advisor_offline.engine.act_safety.act_filter import allowed_cards_for_user
from hybrid_advisor_offline.engine.envs.market_envs import MarketEnv
from hybrid_advisor_offline.engine.personal.personal_prior import (
    build_personal_prior,
    infer_prefs_from_profile,
)
from hybrid_advisor_offline.engine.policy.explain import build_explain_pack
from hybrid_advisor_offline.engine.state.state_builder import (
    MarketSnapshot,
    UserProfile,
    build_state_vec,
)
from hybrid_advisor_offline.offline.trainrl import train_cql
from hybrid_advisor_offline.offline.trainrl.train_cql import load_cql_policy_from_paths


def _predict_q_values(policy, state_vec: np.ndarray) -> np.ndarray:
    act_size = len(ALL_CARDS)
    state_batch = np.repeat(state_vec[None, :], act_size, axis=0)
    action_batch = np.arange(act_size, dtype=np.int64)
    return policy.predict_value(state_batch, action_batch)


class UserProfileInput(BaseModel):
    age: int = Field(ge=18, le=90)
    job: str
    marital: str
    education: str
    default: str
    balance: int = Field(ge=0)
    housing: str
    loan: str
    contact: Optional[str] = "cellular"
    day_of_week: Optional[str] = "mon"
    month: Optional[str] = "may"
    pdays: Optional[int] = -1
    previous: Optional[int] = 0
    poutcome: Optional[str] = "unknown"
    current_alloc: List[float] = Field(
        default_factory=lambda: [0.4, 0.4, 0.2], min_items=3, max_items=3
    )

    @validator("current_alloc")
    def _normalize_alloc(cls, value: List[float]):
        arr = np.asarray(value, dtype=np.float32)
        if np.any(arr < 0):
            raise ValueError("current_alloc 不可为负数")
        total = float(np.sum(arr))
        if total <= 0:
            raise ValueError("current_alloc 之和需大于 0")
        arr = arr / total
        return arr.tolist()


DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
DEMO_DATASET_PATH = os.getenv("DEMO_DATASET_PATH", "./data/offline_dataset_demo.h5")
DEMO_MODEL_PATH = os.getenv("DEMO_MODEL_PATH", "./models/cql_demo.pt")
FULL_DATASET_PATH = os.getenv(
    "STREAMLIT_DATASET_PATH",
    getattr(train_cql, "DATASET_PATH", "./data/offline_dataset.h5"),
)
FULL_MODEL_PATH = os.getenv(
    "STREAMLIT_MODEL_PATH",
    getattr(train_cql, "MODEL_SAVE_PATH", "./models/cql_discrete_model.pt"),
)


class RecommendationResponse(BaseModel):
    chosen_card_id: str
    target_alloc: List[float]
    description: str
    q_value: float
    customer_friendly_text: str
    audit_hash: str
    translator_meta: str


app = FastAPI(
    title="Hybrid Advisor Offline · 影子接口",
    description="离线 RL 策略的只读接口，供审计与影子评测使用。",
    version="0.1.0",
)

_cql_policy = None
_latest_snapshot: Optional[MarketSnapshot] = None


@app.on_event("startup")
def _load_resources():
    global _cql_policy, _latest_snapshot
    try:
        dataset_path = DEMO_DATASET_PATH if DEMO_MODE else FULL_DATASET_PATH
        model_path = DEMO_MODEL_PATH if DEMO_MODE else FULL_MODEL_PATH
        _cql_policy = load_cql_policy_from_paths(
            dataset_path,
            model_path,
            require_gpu=False,
        )
        env = MarketEnv()
        _latest_snapshot = env.mkt_sshots[-1]
        mode = "DEMO" if DEMO_MODE else "FULL"
        print(f"影子接口：模型与市场快照加载成功（{mode}）。")
    except Exception as exc:  # pragma: no cover - 启动日志
        _cql_policy = None
        _latest_snapshot = None
        print(f"影子接口：资源加载失败 {exc}")


@app.get("/", summary="健康检查")
def health_check():
    status = "ready" if _cql_policy and _latest_snapshot is not None else "loading"
    return {"status": status, "message": "Hybrid Advisor Offline API"}


@app.post("/recommend", response_model=RecommendationResponse, summary="获取 AI 推荐")
def recommend(payload: UserProfileInput):
    if _cql_policy is None or _latest_snapshot is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    profile = UserProfile(**payload.dict(exclude={"current_alloc"}))
    current_alloc = np.asarray(payload.current_alloc, dtype=np.float32)
    state_vec = build_state_vec(_latest_snapshot, profile, current_alloc)
    q_values = _predict_q_values(_cql_policy, state_vec)

    allowed_cards = allowed_cards_for_user(profile.risk_bucket)
    allowed_ids = [card.act_id for card in allowed_cards]
    if not allowed_ids:
        raise HTTPException(status_code=422, detail="当前约束下没有可用的动作")

    priors = build_personal_prior(
        allowed_ids,
        prefs=infer_prefs_from_profile(profile),
        risk_bucket=profile.risk_bucket,
    )

    mask = np.full_like(q_values, -np.inf, dtype=np.float32)
    for act_id in allowed_ids:
        if 0 <= act_id < len(mask):
            mask[act_id] = 0.0
    masked_q = q_values + mask

    if priors:
        prior_vec = np.zeros_like(q_values)
        for act_id, bump in priors.items():
            if 0 <= act_id < len(prior_vec):
                prior_vec[act_id] = bump
        masked_q = masked_q + prior_vec
    chosen_id = int(np.argmax(masked_q))
    card = get_card_by_id(chosen_id)
    explain = build_explain_pack(card, profile.risk_bucket)
    customer_text, translator_meta = refine_text(
        explain["customer_friendly_text"],
        {
            "card_id": card.card_id,
            "card_risk_level": card.risk_level,
            "user_risk_bucket": profile.risk_bucket,
            "target_alloc": card.target_alloc,
        },
    )
    audit_hash = hashlib.sha256(explain["audit_text"].encode("utf-8")).hexdigest()

    return RecommendationResponse(
        chosen_card_id=card.card_id,
        target_alloc=card.target_alloc,
        description=card.description,
        q_value=float(masked_q[chosen_id]),
        customer_friendly_text=customer_text,
        audit_hash=audit_hash,
        translator_meta=translator_meta,
    )


if __name__ == "__main__":  # pragma: no cover - 手动运行
    import uvicorn

    uvicorn.run("hybrid_advisor_offline.ux.shadow_api:app", host="0.0.0.0", port=8000, reload=True)
