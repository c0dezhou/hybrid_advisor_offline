from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence

from hybrid_advisor_offline.engine.act_safety.act_card_def import ActCard
from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import (
    get_card_by_id,
)

# 个性化 prior 的全部开关与力度，一股脑集中在这里，方便灰度
USE_PERSONAL_PRIOR = int(os.environ.get("USE_PERSONAL_PRIOR", "1")) == 1
PRIOR_CAP = float(os.environ.get("PRIOR_CAP", "0.08"))
RISK_BUCKET_PUSH = float(os.environ.get("PRIOR_BUCKET_PUSH", "0.015"))
RISK_HINT_PUSH = float(os.environ.get("PRIOR_HINT_PUSH", "0.02"))
HORIZON_PUSH = float(os.environ.get("PRIOR_HORIZON_PUSH", "0.01"))
EQUITY_CAP_PUSH = float(os.environ.get("PRIOR_EQUITY_CAP_PUSH", "0.03"))

_RISK_HINT_MAP = {
    0: "conservative",
    1: "moderate",
    2: "aggressive",
}


def _collect_cards(allowed: Sequence[int | ActCard]) -> list[ActCard]:
    """把 act_id 或 ActCard 统一整理成 ActCard 列表，容错一些奇怪输入。"""
    card_list: list[ActCard] = []
    for item in allowed:
        if isinstance(item, ActCard):
            card_list.append(item)
            continue
        try:
            card_list.append(get_card_by_id(int(item)))
        except (KeyError, ValueError, TypeError):
            continue
    return card_list


def _normalize_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def build_personal_prior(
    allowed_ids: Sequence[int | ActCard],
    *,
    prefs: Mapping[str, Any] | None = None,
    risk_bucket: int | None = None,
):
    """
    给被允许的动作卡片打一层“人工偏置”，表达业务规则中对不同客户的偏好。
    - 当 USE_PERSONAL_PRIOR=0：直接返回全 0，方便做对照实验；
    - 当开关开启：综合 risk bucket、口头偏好、投资期限、股权上限来炼制 bump。
    """
    cards = _collect_cards(allowed_ids)
    if not cards:
        return {}

    if not USE_PERSONAL_PRIOR:
        return {card.act_id: 0.0 for card in cards}

    prefs = prefs or {}
    risk_hint = str(
        prefs.get("risk_hint")
        or prefs.get("risk_tolerance_hint")
        or ""
    ).lower()
    bucket_anchor = (
        risk_bucket
        if risk_bucket is not None
        else prefs.get("risk_bucket", max(card.risk_level for card in cards))
    )

    horizon_years = prefs.get("horizon_years")
    equity_cap = _normalize_float(prefs.get("equity_cap") or prefs.get("max_equity"))

    priors: Dict[int, float] = {}
    for card in cards:
        bump_val = 0.0

        # 1) 风险等级约束：越接近 bucket 上限越加分
        risk_delta = card.risk_level - bucket_anchor
        bump_val += risk_delta * RISK_BUCKET_PUSH

        # 2) risk_hint：一句话偏好，直接加一个方向性的倾斜
        if risk_hint in {"aggressive", "进取", "high"}:
            bump_val += card.risk_level * RISK_HINT_PUSH
        elif risk_hint in {"conservative", "保守", "cautious", "low"}:
            bump_val -= card.risk_level * RISK_HINT_PUSH

        # 3) 投资期限：长线=偏股，短线=偏债/现金
        if isinstance(horizon_years, (int, float)):
            equity_bias = card.target_alloc[0] - 0.4  # 以 40% 股票为中性
            if horizon_years >= 7:
                bump_val += equity_bias * HORIZON_PUSH
            elif horizon_years <= 3:
                bump_val -= equity_bias * HORIZON_PUSH

        # 4) 股权上限：超过用户设定的最大股票仓位就扣分
        if equity_cap is not None and card.target_alloc[0] > equity_cap:
            bump_val -= (card.target_alloc[0] - equity_cap) * EQUITY_CAP_PUSH

        # 5) 最后做 clip，保证 prior 只是一点“推力”，不会盖过模型
        priors[card.act_id] = float(max(-PRIOR_CAP, min(bump_val, PRIOR_CAP)))

    return priors


def infer_prefs_from_profile(user_profile: Any):
    """
    根据 UCI 用户画像推断出 rule-based 偏好，供 build_personal_prior 复用。
    - 年轻 / 高余额 → 更激进；
    - 高龄 / 有贷款 → 更保守；
    - 年龄映射到 horizon_years。
    """
    if user_profile is None:
        return {}

    bucket = getattr(user_profile, "risk_bucket", None)
    risk_hint = _RISK_HINT_MAP.get(bucket, "moderate")

    age = getattr(user_profile, "age", None)
    housing = str(getattr(user_profile, "housing", "")).lower()
    loan = str(getattr(user_profile, "loan", "")).lower()

    if isinstance(age, (int, float)):
        if age <= 30:
            horizon_years = 10
        elif age >= 55:
            horizon_years = 3
        else:
            horizon_years = 5
    else:
        horizon_years = 5

    balance = getattr(user_profile, "balance", None)
    if isinstance(balance, (int, float)) and balance >= 20000:
        risk_hint = "aggressive"
    if any(flag == "yes" for flag in (housing, loan)):
        risk_hint = "conservative"

    return {
        "risk_hint": risk_hint,
        "horizon_years": horizon_years,
        "risk_bucket": bucket,
    }
