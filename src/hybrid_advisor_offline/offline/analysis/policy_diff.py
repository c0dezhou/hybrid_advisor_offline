from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from d3rlpy.dataset import ReplayBuffer

from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_card_by_id
from hybrid_advisor_offline.engine.act_safety.act_filter import allowed_cards_for_user
from hybrid_advisor_offline.engine.policy import policy_based_rule
from hybrid_advisor_offline.engine.state.state_builder import UserProfile
from hybrid_advisor_offline.offline.analysis.segment_metrics import (
    _npz_to_profiles,
    describe_user_segments,
)
from hybrid_advisor_offline.offline.eval.policy_loader import load_trained_policy


def _load_buffer(path: str) -> ReplayBuffer:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据集：{path}")
    return ReplayBuffer.load(path)


def _select_scenarios(
    buffer: ReplayBuffer,
    num_cases: int,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    picks: List[Tuple[int, int]] = []
    all_indices = []
    for ep_idx, episode in enumerate(buffer.episodes):
        length = len(episode.observations)
        for step_idx in range(length):
            all_indices.append((ep_idx, step_idx))
    if not all_indices:
        raise ValueError("ReplayBuffer 中没有可用的状态。")
    num_cases = min(num_cases, len(all_indices))
    chosen = rng.choice(len(all_indices), size=num_cases, replace=False)
    for idx in chosen:
        picks.append(all_indices[idx])
    return picks


def _format_card_view(act_id: int) -> Dict[str, object]:
    card = get_card_by_id(act_id)
    return {
        "action_id": act_id,
        "card_id": card.card_id,
        "risk_level": card.risk_level,
        "equity_weight": float(card.target_alloc[0]),
        "target_alloc": card.target_alloc,
    }


def _apply_rule_policy(state_vec, profile: UserProfile):
    allowed_cards = allowed_cards_for_user(profile.risk_bucket)
    action_id, _ = policy_based_rule.policy_based_rule(
        state_vec,
        allowed_cards,
        profile.risk_bucket,
    )
    return action_id


def _predict_action(policy, state_vec: np.ndarray) -> int:
    action = policy.predict(state_vec[None, :])
    return int(action[0])


def _build_case_payload(
    profile: UserProfile,
    state_vec: np.ndarray,
    seg_key: str,
    step_idx: int,
    rule_action: int,
    bc_action: Optional[int],
    bcq_action: Optional[int],
) -> Dict[str, object]:
    user_view = {
        "age": profile.age,
        "balance": profile.balance,
        "risk_bucket": profile.risk_bucket,
        "housing": profile.housing,
        "loan": profile.loan,
    }
    payload = {
        "user_segment": seg_key,
        "user_profile": user_view,
        "state_step": step_idx,
        "rule": _format_card_view(rule_action),
    }
    if bc_action is not None:
        payload["bc"] = _format_card_view(bc_action)
    if bcq_action is not None:
        payload["bcq"] = _format_card_view(bcq_action)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比规则策略与 BC/BCQ 推荐差异，并导出 JSON。",
    )
    parser.add_argument("--dataset", required=True, help="ReplayBuffer .h5 路径。")
    parser.add_argument(
        "--behavior-meta",
        required=True,
        help="包含 user_profiles 的 npz 文件，用于还原用户画像。",
    )
    parser.add_argument("--bc-model", default=None, help="BC 模型权重路径。")
    parser.add_argument("--bcq-model", default=None, help="BCQ 模型权重路径。")
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=50,
        help="抽取多少个状态场景做对比（默认 50）。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSON 路径。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机抽样种子。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buffer = _load_buffer(args.dataset)
    profiles = _npz_to_profiles(args.behavior_meta)
    if not profiles:
        raise ValueError("user_profiles 为空。")

    bc_policy = (
        load_trained_policy(args.bc_model, buffer, require_gpu=False)
        if args.bc_model
        else None
    )
    bcq_policy = (
        load_trained_policy(args.bcq_model, buffer, require_gpu=False)
        if args.bcq_model
        else None
    )

    picks = _select_scenarios(buffer, args.num_scenarios, seed=args.seed)
    cases: List[Dict[str, object]] = []
    for ep_idx, step_idx in picks:
        episode = buffer.episodes[ep_idx]
        state_vec = np.asarray(episode.observations[step_idx], dtype=np.float32)
        profile_idx = min(ep_idx, len(profiles) - 1)
        profile = profiles[profile_idx]
        seg_key = "|".join([f"{k}={v}" for k, v in describe_user_segments(profile).items()])

        rule_action = _apply_rule_policy(state_vec, profile)
        bc_action = _predict_action(bc_policy, state_vec) if bc_policy else None
        bcq_action = _predict_action(bcq_policy, state_vec) if bcq_policy else None

        cases.append(
            _build_case_payload(
                profile,
                state_vec,
                seg_key,
                step_idx,
                rule_action,
                bc_action,
                bcq_action,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(cases, fp, ensure_ascii=False, indent=2)
    print(f"[policy_diff] 已写入 {output_path}，共 {len(cases)} 条场景。")


if __name__ == "__main__":
    main()
