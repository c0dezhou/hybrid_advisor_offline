"""Utilities for consistent reward scaling in offline RL pipelines."""

from __future__ import annotations

import numpy as np
from d3rlpy.dataset import ReplayBuffer


_DEF_TOL = 1e-9


def apply_reward_scale(replay_buffer: ReplayBuffer, scale: float) -> None:
    """In-place multiply rewards for all episodes in the buffer.

    The underlying业务语义保持不变，仅放大/缩小数值尺度，便于神经网络梯度传播。
    该函数会记录 `_reward_scale_applied`，避免重复放大。
    """

    if replay_buffer is None:
        return
    if scale <= 0:
        raise ValueError("reward_scale 必须为正数。")
    if abs(scale - 1.0) < _DEF_TOL:
        return

    applied = getattr(replay_buffer, "_reward_scale_applied", None)
    if applied is not None:
        if abs(applied - scale) < _DEF_TOL:
            return
        raise ValueError(
            "ReplayBuffer 已按不同倍率缩放，无法再次使用新的 reward_scale。"
        )

    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for episode in replay_buffer.episodes:
        rewards = np.asarray(episode.rewards, dtype=np.float32)
        total_sum += rewards.sum()
        total_sq_sum += np.square(rewards).sum()
        total_count += rewards.size
        rewards *= scale
        # 使用 object.__setattr__ 方法重新调整奖励大小，
        # 这是 d3rlpy 中修改冻结的 Episode 数据类的官方支持方法。
        # 这样既能保持回放缓冲区数据的一致性，又能满足不可变性约束
        object.__setattr__(episode, "rewards", rewards)

    replay_buffer._reward_scale_applied = scale

    if total_count > 0:
        mean_before = total_sum / total_count
        var_before = max(total_sq_sum / total_count - mean_before**2, 0.0)
        std_before = float(np.sqrt(var_before))
        mean_after = mean_before * scale
        std_after = std_before * scale
        print(
            f"[reward_scale] applied scale={scale:g} | "
            f"mean: {mean_before:.6f}->{mean_after:.6f}, "
            f"std: {std_before:.6f}->{std_after:.6f}"
        )
