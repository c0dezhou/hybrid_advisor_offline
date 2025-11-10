from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer


from hybrid_advisor_offline.engine.state.state_builder import UserProfile
from hybrid_advisor_offline.offline.eval.policy_loader import load_trained_policy


@dataclass
class EpisodeSnapshot:
    episode_idx: int
    total_reward: float
    length: int
    estimate: Optional[float] = None


def _ensure_dir(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def _load_replay_buffer(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到离线数据集：{path}")
    buffer =  ReplayBuffer.load(path, buffer=InfiniteBuffer())
    if not buffer.episodes:
        raise ValueError("ReplayBuffer 为空，无法做分群统计。")
    return buffer


def load_episode_summaries(summary_path: Optional[str]) -> Optional[List[EpisodeSnapshot]]:
    if not summary_path:
        return None
    if not os.path.exists(summary_path):
        return None
    df = pd.read_csv(summary_path)
    required = {"episode_id", "total_reward", "length"}
    if not required.issubset(df.columns):
        raise ValueError(f"{summary_path} 缺少所需列 {required}")
    snapshots = [
        EpisodeSnapshot(
            int(row.episode_id),
            float(row.total_reward),
            int(row.length),
            None,
        )
        for row in df.itertuples(index=False)
    ]
    return snapshots


def _npz_to_profiles(meta_path: str) -> List[UserProfile]:
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到行为策略 meta：{meta_path}")
    payload = np.load(meta_path, allow_pickle=True)
    profile_key = "user_profiles" if "user_profiles" in payload else None
    if profile_key is None:
        raise KeyError(
            "behavior meta 中缺少 user_profiles。请重新生成数据集，"
            "或在 gen_datasets.py 中确保 np.savez 写入 user_profiles。"
        )
    raw_profiles = payload[profile_key]
    profiles: List[UserProfile] = []
    for item in raw_profiles:
        if isinstance(item, UserProfile):
            profiles.append(item)
        else:
            candidate: Any = item
            if hasattr(candidate, "item"):
                candidate = candidate.item()
            if isinstance(candidate, (bytes, str)):
                candidate = json.loads(candidate)
            profiles.append(UserProfile(**dict(candidate)))
    return profiles


def describe_user_segments(profile: UserProfile) -> Dict[str, str]:
    risk_map = {0: "conservative", 1: "moderate", 2: "aggressive"}
    risk_tag = risk_map.get(getattr(profile, "risk_bucket", None), "unknown")

    age = getattr(profile, "age", 0)
    if age < 30:
        age_tag = "young"
    elif age <= 55:
        age_tag = "middle"
    else:
        age_tag = "senior"

    balance = getattr(profile, "balance", 0)
    if balance < 5_000:
        bal_tag = "low"
    elif balance < 50_000:
        bal_tag = "mid"
    else:
        bal_tag = "high"

    return {
        "risk": risk_tag,
        "age_band": age_tag,
        "balance_band": bal_tag,
    }


def _segment_key(seg: Dict[str, str]) -> str:
    return "|".join(f"{k}={v}" for k, v in seg.items())


def _load_policy(model_path: Optional[str], buffer: ReplayBuffer):
    if not model_path:
        return None
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型：{model_path}")
    return load_trained_policy(model_path, buffer, require_gpu=False)


def _estimate_episode_value(policy, episode) -> float:
    obs = np.asarray(episode.observations, dtype=np.float32)
    if obs.size == 0:
        return 0.0
    greedy_actions = policy.predict(obs)
    q_values = policy.predict_value(obs, greedy_actions)
    if q_values.size == 0:
        return 0.0
    return float(q_values[0])


def collect_episode_snapshots(buffer: ReplayBuffer, policy=None) -> List[EpisodeSnapshot]:
    eps: List[EpisodeSnapshot] = []
    for idx, episode in enumerate(buffer.episodes):
        rewards = np.asarray(episode.rewards, dtype=np.float32)
        total_reward = float(np.sum(rewards))
        length = len(rewards)
        estimate = None
        if policy is not None:
            estimate = _estimate_episode_value(policy, episode)
        eps.append(EpisodeSnapshot(idx, total_reward, length, estimate))
    return eps


def aggregate_by_segment(
    algo_name: str,
    episodes: Iterable[EpisodeSnapshot],
    profiles: List[UserProfile],
) -> pd.DataFrame:
    rows = []
    for snap in episodes:
        try:
            profile = profiles[snap.episode_idx]
        except IndexError:
            continue
        seg_map = describe_user_segments(profile)
        seg_key = _segment_key(seg_map)
        metric_value = snap.estimate if snap.estimate is not None else snap.total_reward
        rows.append(
            {
                "algo_name": algo_name,
                "segment_key": seg_key,
                "episode_idx": snap.episode_idx,
                "metric_value": metric_value,
                "episode_length": snap.length,
            }
        )
    if not rows:
        raise ValueError("没有可用于聚合的 episode 数据。")
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["algo_name", "segment_key"])
        .agg(
            num_episodes=("episode_idx", "count"),
            avg_episode_return=("metric_value", "mean"),
            std_episode_return=("metric_value", "std"),
            avg_episode_length=("episode_length", "mean"),
        )
        .reset_index()
    )
    grouped["std_episode_return"] = grouped["std_episode_return"].fillna(0.0)
    return grouped


def plot_segment_bars(stats: pd.DataFrame, output_path: Optional[Path] = None):
    segs = stats["segment_key"].unique()
    algos = stats["algo_name"].unique()
    seg_positions = np.arange(len(segs), dtype=np.float32)
    bar_width = 0.8 / max(len(algos), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(segs) * 0.6), 4.5))
    for idx, algo in enumerate(algos):
        subset = stats[stats["algo_name"] == algo]
        heights = [subset[subset["segment_key"] == seg]["avg_episode_return"].mean() for seg in segs]
        positions = seg_positions + idx * bar_width - 0.4 + bar_width / 2
        ax.bar(positions, heights, width=bar_width, label=algo)

    ax.set_xticks(seg_positions)
    ax.set_xticklabels(segs, rotation=35, ha="right")
    ax.set_ylabel("Avg Episode Return")
    ax.set_title("Segmented Episode Returns by Algorithm")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=200)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按人群分段统计离线策略表现，并输出 CSV + 图表。",
    )
    parser.add_argument("--dataset", required=True, help="ReplayBuffer .h5 路径。")
    parser.add_argument(
        "--behavior-meta",
        required=True,
        help="行为策略 meta（npz）路径，须包含 user_profiles。",
    )
    parser.add_argument(
        "--episode-summary",
        required=False,
        default=None,
        help="可选的 episode 汇总 CSV，若提供且不加载策略，则直接复用。",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="策略模型路径，若缺省则表示行为策略 baseline。",
    )
    parser.add_argument(
        "--algo-name",
        required=True,
        help="算法标签（如 baseline_behavior / bc_reward_personal）。",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="指标 CSV 输出路径。",
    )
    parser.add_argument(
        "--output-fig",
        required=False,
        default=None,
        help="图表输出路径（PNG）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_episode_summaries(args.episode_summary)
    buffer = None
    if args.model or summaries is None:
        buffer = _load_replay_buffer(args.dataset)

    profiles = _npz_to_profiles(args.behavior_meta)
    expected_episodes = len(summaries) if summaries is not None else len(buffer.episodes)
    if len(profiles) < expected_episodes:
        raise ValueError("user_profiles 数量少于 episode 数，无法对齐。")

    policy = _load_policy(args.model, buffer) if buffer is not None and args.model else None

    if summaries is not None and policy is None:
        snapshots = summaries
    else:
        if buffer is None:
            buffer = _load_replay_buffer(args.dataset)
        snapshots = collect_episode_snapshots(buffer, policy=policy)
    stats = aggregate_by_segment(args.algo_name, snapshots, profiles)

    csv_path = Path(args.output_csv)
    _ensure_dir(csv_path)
    stats.to_csv(csv_path, index=False)
    print(f"[segment_metrics] CSV 已写入 {csv_path}")

    fig_path = Path(args.output_fig) if args.output_fig else None
    if fig_path:
        plot_segment_bars(stats, fig_path)
        print(f"[segment_metrics] 图表已保存至 {fig_path}")


if __name__ == "__main__":
    main()
