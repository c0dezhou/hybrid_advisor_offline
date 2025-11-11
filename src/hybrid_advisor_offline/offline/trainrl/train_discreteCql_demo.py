"""
轻量版 CQL 训练脚本：使用小规模 demo 数据集，快速产出一个可供前端演示的策略。

示例：
    python -m hybrid_advisor_offline.offline.trainrl.train_discrete_demo \
        --dataset ./data/offline_dataset_demo.h5 \
        --model-output ./models/cql_demo.pt \
        --steps 100000 \
        --reward-scale 300 \
        --require-gpu
"""

from __future__ import annotations

import argparse
import json
import os

from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer, create_infinite_replay_buffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.logging import FileAdapterFactory
from sklearn.model_selection import train_test_split

from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_act_space_size
from hybrid_advisor_offline.offline.utils.reward_scaling import apply_reward_scale
from hybrid_advisor_offline.offline.trainrl.train_cql import (
    _require_gpu,
    _standard_dataset,
    ALPHA as FULL_ALPHA,
    LEARNING_RATE as FULL_LR,
    TARGET_UPDATE_INTERVAL,
    N_CRITICS,
    USE_REWARD_SCALER,
)


DEFAULT_DEMO_DATASET = os.getenv("DEMO_DATASET_PATH", "./data/offline_dataset_demo.h5")
DEFAULT_DEMO_MODEL = os.getenv("DEMO_MODEL_PATH", "./models/cql_demo.pt")
DEFAULT_DEMO_STEPS = int(os.getenv("DEMO_CQL_STEPS", "100000"))
DEFAULT_DEMO_STEPS_PER_EPOCH = int(os.getenv("DEMO_CQL_STEPS_PER_EPOCH", "2000"))
DEFAULT_DEMO_REWARD_SCALE = float(os.getenv("DEMO_CQL_REWARD_SCALE", "300.0"))
DEFAULT_BATCH_SIZE = int(os.getenv("DEMO_CQL_BATCH_SIZE", "256"))


def _model_config_path(model_path: str) -> str:
    return f"{model_path}.config.json"


def train_demo_cql(
    dataset_path: str,
    model_path: str,
    *,
    steps: int,
    steps_per_epoch: int,
    alpha: float,
    learning_rate: float,
    reward_scale: float,
    batch_size: int,
    require_gpu: bool,
) -> None:
    if require_gpu:
        _require_gpu()

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到 demo 数据集：{dataset_path}")

    dataset = ReplayBuffer.load(dataset_path, buffer=InfiniteBuffer())
    apply_reward_scale(dataset, reward_scale)
    obs_scaler, rew_scaler = _standard_dataset(
        dataset,
        use_reward_scaler=USE_REWARD_SCALER,
    )

    episodes = list(dataset.episodes)
    if not episodes:
        raise RuntimeError("数据集中没有 episode，无法训练。")
    train_eps, val_eps = train_test_split(episodes, test_size=0.2, random_state=42)
    train_buffer = create_infinite_replay_buffer(episodes=train_eps)
    val_buffer = create_infinite_replay_buffer(episodes=val_eps)

    action_size = get_act_space_size()
    print(
        f"[demo_cql] dataset={dataset_path}, transitions={dataset.size()}, "
        f"train_eps={len(train_eps)}, val_eps={len(val_eps)}, action_size={action_size}"
    )

    config = DiscreteCQLConfig(
        observation_scaler=obs_scaler,
        reward_scaler=rew_scaler,
        alpha=alpha,
        learning_rate=learning_rate,
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        batch_size=batch_size,
    )
    device = 0 if require_gpu else False
    algo = config.create(device=device)

    algo.fit(
        train_buffer,
        n_steps=steps,
        n_steps_per_epoch=steps_per_epoch,
        eval_episodes=val_buffer.episodes,
        logger_adapter=FileAdapterFactory(root_dir="d3rlpy_logs/cql_demo"),
        show_progress=True,
    )

    model_dir = os.path.dirname(model_path) or "."
    os.makedirs(model_dir, exist_ok=True)
    algo.save_model(model_path)
    cfg_payload = {
        "alpha": alpha,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_critics": N_CRITICS,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "n_steps": steps,
        "n_steps_per_epoch": steps_per_epoch,
        "reward_scale": reward_scale,
        "use_reward_scaler": USE_REWARD_SCALER,
        "dataset_path": dataset_path,
    }
    cfg_path = _model_config_path(model_path)
    with open(cfg_path, "w", encoding="utf-8") as fp:
        json.dump(cfg_payload, fp, ensure_ascii=False, indent=2)
    print(f"[demo_cql] 模型已保存至 {model_path}，配置写入 {cfg_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 demo 版 Discrete CQL。")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DEMO_DATASET,
        help=f"demo 数据集路径（默认 {DEFAULT_DEMO_DATASET}）。",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=DEFAULT_DEMO_MODEL,
        help=f"模型保存路径（默认 {DEFAULT_DEMO_MODEL}）。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_DEMO_STEPS,
        help=f"训练步数（默认 {DEFAULT_DEMO_STEPS}）。",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=DEFAULT_DEMO_STEPS_PER_EPOCH,
        help=f"每个 epoch 的步数（默认 {DEFAULT_DEMO_STEPS_PER_EPOCH}）。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=FULL_ALPHA,
        help=f"CQL 保守系数（默认 {FULL_ALPHA}）。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=FULL_LR,
        help=f"学习率（默认 {FULL_LR}）。",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=DEFAULT_DEMO_REWARD_SCALE,
        help=f"reward scale（默认 {DEFAULT_DEMO_REWARD_SCALE}）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"batch size（默认 {DEFAULT_BATCH_SIZE}）。",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="若指定则要求使用 GPU。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_demo_cql(
        dataset_path=args.dataset,
        model_path=args.model_output,
        steps=max(1, args.steps),
        steps_per_epoch=max(1, args.steps_per_epoch),
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        reward_scale=args.reward_scale,
        batch_size=max(1, args.batch_size),
        require_gpu=args.require_gpu,
    )


if __name__ == "__main__":
    main()
