"""
从完整的离线数据集中抽样出一个轻量 demo 版本，便于前端快速载入。

示例：
    python -m hybrid_advisor_offline.offline.trainrl.make_demo_dataset \
        --source ./data/offline_dataset_reward_personal.h5 \
        --target ./data/offline_dataset_demo.h5 \
        --n-samples 10000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.dataset import Episode


DEFAULT_SOURCE_DATASET = os.getenv(
    "DEMO_SOURCE_DATASET",
    "./data/offline_dataset_reward_personal.h5",
)
DEFAULT_TARGET_DATASET = os.getenv(
    "DEMO_TARGET_DATASET",
    "./data/offline_dataset_demo.h5",
)
DEFAULT_N_SAMPLES = int(os.getenv("DEMO_DATASET_TRANSITIONS", "10000"))


def _behavior_meta_path(dataset_path: str | Path) -> Path:
    dataset_path = Path(dataset_path)
    stem = dataset_path.with_suffix("")
    return stem.with_name(f"{stem.name}_behavior.npz")


def _select_episode_indices(
    episodes: Sequence[Episode],
    target_transitions: int,
    seed: int,
) -> Tuple[List[int], int]:
    """随机选择若干 episode，使得其 transition 总数 >= target_transitions。"""
    lengths = np.array([len(ep) for ep in episodes], dtype=np.int64)
    total = lengths.sum()
    if target_transitions <= 0 or target_transitions >= total:
        return list(range(len(episodes))), int(total)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    picked: List[int] = []
    acc = 0
    for idx in order:
        picked.append(int(idx))
        acc += int(lengths[idx])
        if acc >= target_transitions:
            break
    picked.sort()
    return picked, acc


def _build_transition_mask(
    episodes: Sequence[Episode],
    selected_indices: Sequence[int],
) -> np.ndarray:
    """生成一个 bool mask，标记被保留的 transition 索引。"""
    selected = set(selected_indices)
    total_transitions = sum(len(ep) for ep in episodes)
    mask = np.zeros(total_transitions, dtype=bool)
    cursor = 0
    for idx, ep in enumerate(episodes):
        length = len(ep)
        if idx in selected:
            mask[cursor : cursor + length] = True
        cursor += length
    return mask


def _subset_behavior_meta_safe(
    source_meta: Path,
    target_meta: Path,
    episodes: Sequence[Episode],
    selected_indices: List[int],
    transition_mask: np.ndarray,
) -> None:
    if not source_meta.exists():
        print(f"[demo_dataset] 行为策略 meta 未找到，跳过：{source_meta}")
        return

    data = np.load(source_meta, allow_pickle=True)
    total_transitions = transition_mask.size
    episode_count = (
        data["user_profiles"].shape[0]
        if "user_profiles" in data.files
        else len(episodes)
    )

    episode_mask = np.zeros(episode_count, dtype=bool)
    for idx in selected_indices:
        if idx < episode_count:
            episode_mask[idx] = True
    id_map = {old_id: new_id for new_id, old_id in enumerate(selected_indices)}

    out: Dict[str, np.ndarray] = {}
    for key in data.files:
        arr = data[key]
        if arr.shape[0] == total_transitions:
            filtered = arr[transition_mask]
            if key == "episode_ids":
                filtered = np.array([id_map[int(old)] for old in filtered], dtype=np.int32)
            out[key] = filtered
        elif arr.shape[0] == episode_count:
            filtered = arr[episode_mask]
            out[key] = filtered
        else:
            out[key] = arr

    target_meta.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target_meta, **out)
    print(f"[demo_dataset] 行为 meta 已保存至 {target_meta}")


def make_demo_dataset(
    source: Path,
    target: Path,
    *,
    n_samples: int,
    seed: int,
    behavior_meta: Path | None = None,
) -> None:
    if not source.exists():
        raise FileNotFoundError(f"找不到源数据集：{source}")

    print(f"[demo_dataset] 载入 {source} ...")
    buffer = ReplayBuffer.load(str(source), buffer=InfiniteBuffer())
    episodes = list(buffer.episodes)
    total_transitions = sum(len(ep) for ep in episodes)
    selected_indices, kept = _select_episode_indices(episodes, n_samples, seed)
    if not selected_indices:
        raise RuntimeError("未能抽样到任何 episode，请检查 n_samples 参数。")

    print(
        f"[demo_dataset] 共 {len(episodes)} 条 episode，选中 {len(selected_indices)} 条，"
        f"保留 transition {kept}/{total_transitions}。",
    )
    demo_buffer = ReplayBuffer(buffer=InfiniteBuffer())
    demo_buffer.dataset_info = buffer.dataset_info
    for idx in selected_indices:
        demo_buffer.append_episode(episodes[idx])

    target.parent.mkdir(parents=True, exist_ok=True)
    demo_buffer.dump(str(target))
    print(f"[demo_dataset] 已写入 demo 数据集：{target}")

    if behavior_meta is None:
        behavior_meta = _behavior_meta_path(source)
    target_meta = _behavior_meta_path(target)
    transition_mask = _build_transition_mask(episodes, selected_indices)
    _subset_behavior_meta_safe(
        behavior_meta,
        target_meta,
        episodes,
        selected_indices,
        transition_mask,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从完整离线数据集中抽样生成 demo 数据。")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(DEFAULT_SOURCE_DATASET),
        help=f"源数据集路径（默认 {DEFAULT_SOURCE_DATASET}）。",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(DEFAULT_TARGET_DATASET),
        help=f"目标 demo 数据集路径（默认 {DEFAULT_TARGET_DATASET}）。",
    )
    parser.add_argument(
        "--behavior-meta",
        type=Path,
        default=None,
        help="源行为策略 meta（默认为 source 同名 + _behavior.npz）。",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="希望保留的 transition 数（默认 10000）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于 episode 抽样（默认 42）。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    make_demo_dataset(
        args.source,
        args.target,
        n_samples=max(1, args.n_samples),
        seed=args.seed,
        behavior_meta=args.behavior_meta,
    )


if __name__ == "__main__":
    main()
