import numpy as np
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import Episode, MDPDataset, ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

from hybrid_advisor_offline.offline.eval.cpe_metrics import compute_cpe_report
from hybrid_advisor_offline.offline.eval.fqe_data import (
    load_replay_buffer,
    prepare_fqe_datasets,
)
from hybrid_advisor_offline.offline.eval.fqe_runner import run_fqe
from hybrid_advisor_offline.offline.eval.policy_loader import (
    build_scalers,
    load_trained_policy,
)


def _build_toy_episode():
    """
    构造一个简单的两步轨迹，奖励恒为正，动作空间为离散 {0,1}。
    """
    observations = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    actions = np.array([0, 0], dtype=np.int64)
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    return Episode(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=True,
    )


def _build_toy_dataset():
    """
    与 Episode 对应的 MDPDataset，供离线训练使用。
    """
    observations = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    actions = np.array([0, 0], dtype=np.int64)
    rewards = np.array([1.0, 1.0], dtype=np.float32)
    terminals = np.array([0.0, 1.0], dtype=np.float32)
    return MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        action_space=ActionSpace.DISCRETE,
        action_size=2,
    )


def test_eval_pipeline_produces_positive_value(tmp_path):
    """
    验证整个评估流程：训练一个玩具策略，保存后再次加载，
    确认 FQE 与 CPE 指标能够正确返回正收益。
    """
    # 构建离线数据并保存为 ReplayBuffer（评估阶段会直接读取 .h5）
    episode = _build_toy_episode()
    replay_buffer = ReplayBuffer(
        buffer=InfiniteBuffer(),
        episodes=[episode],
        action_space=ActionSpace.DISCRETE,
        action_size=2,
    )
    dataset_path = tmp_path / "offline_dataset.h5"
    replay_buffer.dump(str(dataset_path))
    behavior_meta_path = dataset_path.with_name(f"{dataset_path.stem}_behavior.npz")
    total_steps = sum(ep.transition_count for ep in replay_buffer.episodes)
    propensities = np.full(total_steps, 0.5, dtype=np.float32)
    actions = np.zeros(total_steps, dtype=np.int64)
    episode_ids = np.zeros(total_steps, dtype=np.int32)
    terminals = np.zeros(total_steps, dtype=np.float32)
    terminals[-1] = 1.0
    np.savez(
        behavior_meta_path,
        propensities=propensities,
        actions=actions,
        episode_ids=episode_ids,
        terminals=terminals,
    )

    # 构建最小训练数据集并训练一个离散 CQL 策略
    dataset = _build_toy_dataset()
    obs_scaler, rew_scaler = build_scalers(replay_buffer)
    config = DiscreteCQLConfig(
        observation_scaler=obs_scaler,
        reward_scaler=rew_scaler,
    )
    policy = config.create(device=False)
    policy.fit(
        dataset,
        n_steps=200,
        n_steps_per_epoch=100,
        show_progress=False,
    )
    model_path = tmp_path / "cql_model.pt"
    policy.save_model(str(model_path))

    # 评估流程：加载 ReplayBuffer、策略，以及运行 FQE 和 CPE
    loaded_buffer = load_replay_buffer(str(dataset_path))
    loaded_policy = load_trained_policy(
        str(model_path),
        loaded_buffer,
        require_gpu=False,
        action_size=2,
    )
    train_ds, _ = prepare_fqe_datasets(
        loaded_buffer,
        validation_ratio=0.0,
    )
    fqe_metrics = run_fqe(
        loaded_policy,
        train_ds,
        val_dataset=None,
        n_steps=500,
        eval_interval=100,
        log_dir=str(tmp_path / "fqe_logs"),
        require_gpu=False,
    )

    cpe_metrics = compute_cpe_report(
        loaded_buffer,
        behavior_meta_path=str(behavior_meta_path),
    )

    assert fqe_metrics["train_initial_state_value"] > 0.5
    assert fqe_metrics["train_average_value"] > 0.5
    assert fqe_metrics["train_avg_reward_per_step"] > 0.0
    assert cpe_metrics["episode_return_mean"] > 0.5
    assert cpe_metrics["ips"] > 0.5
