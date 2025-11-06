# 加载 ReplayBuffer 并按随机拆分生成 FQE 训练/验证 MDPDataset

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import MDPDataset, ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

DATASET_PATH_DEFAULT = "./data/offline_dataset.h5"

def load_replay_buffer(path: str = DATASET_PATH_DEFAULT) -> ReplayBuffer:
    """
    加载存储市场轨迹的离线 ReplayBuffer。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到离线数据集 {path}，请先运行数据生成脚本。"
        )
    return ReplayBuffer.load(path, buffer=InfiniteBuffer())


def _episodes_to_arrays(episodes: Sequence) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将轨迹中零散的每一步（transition）数据（状态、动作、奖励、结束标志）提取出来，
    分别整理并打包成四个独立的 np 数组->即d3rlpy训练所需的标准格式
    """
    obss: List[np.ndarray] = []
    acts: List[int] = []
    rwds: List[float] = []
    dones: List[float] = [] # 标记轨迹是否在此结束

    for episode in episodes:
        ep_obs = np.asarray(episode.observations, dtype=np.float32)
        ep_actions = np.asarray(episode.actions, dtype=np.int64).reshape(-1)
        ep_rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)

        for idx, action in enumerate(ep_actions):
            obss.append(ep_obs[idx])
            acts.append(int(action))
            rwds.append(float(ep_rewards[idx]))
            is_last = (idx == len(ep_actions) - 1) and bool(episode.terminated)
            dones.append(1.0 if is_last else 0.0)

    if not obss:
        raise ValueError("给定的 轨迹（episode） 列表为空，无法构建 MDPDataset。")

    # 忽略next_observation d3rlpy 的 MDPDataset 对象用这种方式（提供 obss, acts, rwds, dones 四个数组）
    # 来构建数据集时，d3rlpy 会自动地、隐式地处理下一个状态
    observations = np.stack(obss)
    actions = np.asarray(acts, dtype=np.int64)
    rewards = np.asarray(rwds, dtype=np.float32)
    terminals = np.asarray(dones, dtype=np.float32)
    return observations, actions, rewards, terminals


def prepare_fqe_datasets(
    replay_buffer: ReplayBuffer,
    validation_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[MDPDataset, Optional[MDPDataset]]:
    """
    将 ReplayBuffer 拆分为 FQE 用的训练集和验证集。
    """
    episodes = list(replay_buffer.episodes)
    if not episodes:
        raise ValueError("ReplayBuffer 中没有 episode，请检查数据生成流程。")

    rng = np.random.default_rng(seed=random_state) #随机数生成器
    rng.shuffle(episodes) #原地将 episodes 列表的顺序完全打乱

    if validation_ratio <= 0 or len(episodes) == 1: # 处理边缘情况
        train_obs, train_act, train_rew, train_term = _episodes_to_arrays(episodes)
        action_size = int(np.max(train_act)) + 1
        train_dataset = MDPDataset(
            observations=train_obs,
            actions=train_act,
            rewards=train_rew,
            terminals=train_term,
            action_space=ActionSpace.DISCRETE,
            action_size=action_size,
        )
        return train_dataset, None

    # 计算切分点：根据总轨迹数量和验证集比例，计算出应该分给验证集的轨迹数量。
    # 有 500 条轨迹，比例为 0.1，则 split_idx 为 50。max(1, ...) 确保即使比例很小，也至少会有一条轨迹被分到验证集。
    split_idx = max(1, int(len(episodes) * validation_ratio))
    val_eps = episodes[:split_idx]
    train_eps = episodes[split_idx:]

    # 打包数据
    train_obs, train_act, train_rew, train_term = _episodes_to_arrays(train_eps)
    train_action_size = int(np.max(train_act)) + 1
    train_dataset = MDPDataset(
        observations=train_obs,
        actions=train_act,
        rewards=train_rew,
        terminals=train_term,
        action_space=ActionSpace.DISCRETE,
        action_size=train_action_size,
    )

    val_obs, val_act, val_rew, val_term = _episodes_to_arrays(val_eps)
    val_action_size = int(np.max(val_act)) + 1
    val_dataset = MDPDataset(
        observations=val_obs,
        actions=val_act,
        rewards=val_rew,
        terminals=val_term,
        action_space=ActionSpace.DISCRETE,
        action_size=val_action_size,
    )

    return train_dataset, val_dataset
