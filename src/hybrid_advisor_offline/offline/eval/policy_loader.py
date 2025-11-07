# 加载已训练的 CQL 策略，并复用训练时的 scaler。
import os,sys
from typing import Optional, Tuple

import numpy as np

from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset, ReplayBuffer
from d3rlpy.preprocessing.observation_scalers import StandardObservationScaler
from d3rlpy.preprocessing.reward_scalers import StandardRewardScaler


def build_scalers(
    replay_buffer: ReplayBuffer,
):
    """
    基于离线数据集拟合观测与奖励标准化器。
    """
    obs_scaler = StandardObservationScaler()
    obs_scaler.fit_with_transition_picker(
        replay_buffer.episodes,
        replay_buffer.transition_picker,
    )

    rew_scaler = StandardRewardScaler()
    rew_scaler.fit_with_transition_picker(
        replay_buffer.episodes,
        replay_buffer.transition_picker,
    )
    return obs_scaler, rew_scaler


def _buffer_to_dataset(
    replay_buffer: ReplayBuffer,
    action_size: Optional[int],
) -> MDPDataset:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    terminals: list[float] = []

    for episode in replay_buffer.episodes:
        ep_obs = np.asarray(episode.observations, dtype=np.float32)
        ep_actions = np.asarray(episode.actions, dtype=np.int64).reshape(-1)
        ep_rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)

        for idx, action in enumerate(ep_actions):
            observations.append(ep_obs[idx])
            actions.append(int(action))
            rewards.append(float(ep_rewards[idx]))
            is_last = (idx == len(ep_actions) - 1) and bool(episode.terminated)
            terminals.append(1.0 if is_last else 0.0)

    if not observations:
        raise ValueError("ReplayBuffer 中没有可用的 transition，无法构建策略。")

    if action_size is None:
        if not actions:
            raise ValueError("无法从 ReplayBuffer 推断动作空间大小，请显式传入 action_size。")
        action_size = int(np.max(np.asarray(actions, dtype=np.int64))) + 1

    return MDPDataset(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
        action_space=ActionSpace.DISCRETE,
        action_size=action_size,
    )


def load_trained_policy(
    model_path: str,
    replay_buffer: ReplayBuffer,
    require_gpu: bool = False,
    action_size: Optional[int] = None,
) -> DiscreteCQL:
    """
    加载已训练好的离散 CQL 策略，并复用与训练阶段一致的 scaler。
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到已训练的策略模型：{model_path}"
        )

    obs_scaler, rew_scaler = build_scalers(replay_buffer)

    config = DiscreteCQLConfig(
        observation_scaler=obs_scaler,
        reward_scaler=rew_scaler,
    )
    device = 0 if require_gpu else False
    policy = config.create(device=device)

    if not replay_buffer.episodes:
        raise ValueError("ReplayBuffer 中没有 transition，无法构建策略。")

    dataset_info = getattr(replay_buffer, "dataset_info", None)
    dataset_action_size = getattr(dataset_info, "action_size", None)

    resolved_action_size = action_size if action_size is not None else dataset_action_size
    if resolved_action_size is None:
        raise ValueError("无法推断动作空间大小，请通过 action_size 参数显式指定。")

    if (
        action_size is not None
        and dataset_action_size is not None
        and action_size != dataset_action_size
    ):
        print(
            f"警告：传入的 action_size={action_size} 与数据集记录的 "
            f"action_size={dataset_action_size} 不一致，将优先使用显式传入的值。",
            file=sys.stderr,
        )

    dataset_for_build = _buffer_to_dataset(replay_buffer, resolved_action_size)
    policy.build_with_dataset(dataset_for_build)
    policy.load_model(model_path)
    return policy
