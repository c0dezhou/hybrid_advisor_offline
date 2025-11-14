#!/usr/bin/env python
"""补齐旧版 `_behavior.npz` 中的缺失字段，并且提供 `target_action_probs`。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from hybrid_advisor_offline.engine.act_safety.act_filter import allowed_cards_for_user
from hybrid_advisor_offline.engine.state.state_builder import UserProfile

_PROFILE_FIELDS = set(UserProfile.__dataclass_fields__)


def _normalize_profile_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {k: raw[k] for k in _PROFILE_FIELDS if k in raw}


def _load_profiles(raw: np.ndarray) -> Dict[int, UserProfile]:
    profiles: Dict[int, UserProfile] = {}
    for idx, item in enumerate(raw):
        value = item
        if isinstance(value, bytes):
            value = value.decode()
        if isinstance(value, str):
            import json

            value = json.loads(value)
        if isinstance(value, np.ndarray) and value.dtype == np.object_:
            value = value.item()
        if isinstance(value, dict):
            filtered = _normalize_profile_dict(value)
            profiles[idx] = UserProfile(**filtered)
            continue
        if isinstance(value, UserProfile):
            profiles[idx] = value
            continue
        raise ValueError(f"无法解析 user_profiles[{idx}]：{type(value)}")
    return profiles


def _infer_allowed_counts(episode_ids: np.ndarray, profiles: Dict[int, UserProfile]) -> np.ndarray:
    counts = np.zeros_like(episode_ids, dtype=np.int16)
    for ep_id in np.unique(episode_ids):
        profile = profiles[int(ep_id)]
        counts[episode_ids == ep_id] = len(allowed_cards_for_user(profile.risk_bucket))
    return counts


def _infer_target_probs(allowed_counts: np.ndarray) -> np.ndarray:
    target_probs = np.zeros_like(allowed_counts, dtype=np.float32)
    mask = allowed_counts > 0
    target_probs[mask] = 1.0 / allowed_counts[mask]
    return target_probs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="修复行为策略 meta 文件")
    parser.add_argument("--behavior-meta", required=True, help="原始 _behavior.npz 路径")
    parser.add_argument("--episode-steps", type=int, default=None, help="可选的轨迹步数，缺省则自动推断")
    parser.add_argument("--output", help="输出路径，默认为覆盖原文件")
    parser.add_argument("--rule-eps", type=float, default=None, help="行为策略当时的 RULE_POLICY_EPS")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    src = Path(args.behavior_meta)
    if not src.exists():
        raise SystemExit(f"行为 meta 不存在：{src}")
    data = dict(np.load(src, allow_pickle=True))

    episode_ids = data["episode_ids"].astype(np.int64)
    profiles = _load_profiles(data["user_profiles"])  # type: ignore[arg-type]

    if "allowed_action_counts" not in data:
        data["allowed_action_counts"] = _infer_allowed_counts(episode_ids, profiles)

    if args.rule_eps is not None:
        eps = float(args.rule_eps)
    else:
        eps = float(
            data.get("rule_eps", np.array([os.getenv("RULE_POLICY_EPS", "0.0")], dtype=np.float32))[0]
        )
    data["rule_eps"] = np.array([eps], dtype=np.float32)
    if "action_sources" not in data:
        propensities = data["propensities"].astype(np.float32)
        sources = np.zeros_like(propensities, dtype=np.int8)
        if eps > 0:
            counts = data["allowed_action_counts"].astype(np.float32)
            uniform = np.clip(eps / np.maximum(counts, 1.0), 0.0, 1.0)
            greedy = np.clip(1.0 - eps + uniform, 0.0, 1.0)
            greedy_mask = np.isclose(propensities, greedy, atol=1e-4)
            sources[~greedy_mask] = 1
            sources[greedy_mask] = 0
        data["action_sources"] = sources

    if "behavior_propensity" not in data:
        sources = data["action_sources"].astype(np.int8)
        counts = data["allowed_action_counts"].astype(np.float32)
        prop = np.zeros_like(counts)
        uniform = np.zeros_like(counts)
        mask = counts > 0
        uniform[mask] = 1.0 / counts[mask]
        prop[sources == 0] = (1.0 - eps) + uniform[sources == 0]
        prop[sources == 1] = uniform[sources == 1]
        prop[sources > 1] = uniform[sources > 1]
        data["behavior_propensity"] = prop.astype(np.float32)

    if "target_action_probs" not in data:
        data["target_action_probs"] = _infer_target_probs(data["allowed_action_counts"])

    if "episode_steps" not in data:
        if args.episode_steps is not None:
            data["episode_steps"] = np.array([args.episode_steps], dtype=np.int32)
        else:
            counts = np.bincount(episode_ids)
            inferred = int(counts.max()) if counts.size else 0
            data["episode_steps"] = np.array([inferred], dtype=np.int32)

    out = Path(args.output) if args.output else src
    np.savez(out, **data)
    print(f"已写入修复文件：{out}")


if __name__ == "__main__":
    main()
