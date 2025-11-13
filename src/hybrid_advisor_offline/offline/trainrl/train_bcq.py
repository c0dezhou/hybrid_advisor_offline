"""离线智能投顾 Demo：BCQ 实验管线训练入口。

该脚本与现有的 DiscreteBC / DiscreteCQL 保持一致的超参与日志风格，
用于训练一个基于 d3rlpy 的离散版 BCQ 策略，并在 config.json 中记录 reward_scale 等信息。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import torch
from d3rlpy.algos import DiscreteBCQConfig
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

from hybrid_advisor_offline.offline.utils.reward_scaling import apply_reward_scale

# 一组默认路径和超参，方便直接开箱
DEFAULT_DATASET = "./data/offline_dataset.h5"
DEFAULT_MODEL_OUTPUT = "./models/bcq_discrete_model.pt"
DEFAULT_STEPS = 500_000
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_REWARD_SCALE = float(os.getenv("BCQ_REWARD_SCALE", "1000.0"))
STEPS_PER_EPOCH = int(os.getenv("BCQ_STEPS_PER_EPOCH", "5000"))
EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "full").lower()
_FAST_MODE_NAMES = {"fast", "dev"}
FAST_BCQ_STEP_CAP = int(os.getenv("BCQ_FAST_STEP_CAP", "200000"))


def _maybe_patch_custom_factories() -> str | None:
    """如果设置了 USE_CARD_FACTORY，就把自定义网络塞进 d3rlpy。"""

    if os.getenv("USE_CARD_FACTORY", "0") != "1":
        return None

    try:
        from hybrid_advisor_offline.offline.utils.network_factories import FCHW
        import d3rlpy.models.q_functions as q_functions
    except ImportError:
        print("警告：USE_CARD_FACTORY=1，但未找到自定义 FCHW，跳过自定义网络。")
        return None

    q_functions.FCHW = FCHW
    print("已启用自定义 FCHW 网络工厂。")
    return "fchw"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BCQ 实验管线：训练 DiscreteBCQ 策略，用于对比行为策略 / BC / CQL。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"离线数据集路径（默认 {DEFAULT_DATASET}）。",
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default=DEFAULT_MODEL_OUTPUT,
        help=f"BCQ 模型权重输出路径（默认 {DEFAULT_MODEL_OUTPUT}）。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"训练步数（默认 {DEFAULT_STEPS}）。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help=f"优化器学习率（默认 {DEFAULT_LR}）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"批大小（默认 {DEFAULT_BATCH_SIZE}）。",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=DEFAULT_REWARD_SCALE,
        help=f"离线奖励缩放倍率（默认 {DEFAULT_REWARD_SCALE}）。",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="若指定，则必须检测到 GPU，否则报错。",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="快速实验模式：缩短训练步数，方便调参。",
    )
    return parser.parse_args()


def _prepare_device(require_gpu: bool) -> Tuple[int | bool, str]:
    """根据 --require-gpu 参数返回 d3rlpy 需要的 use_gpu 标志和人类可读的设备名。"""

    if require_gpu:
        if not torch.cuda.is_available():
            print(
                "未检测到可用 GPU。请确认 CUDA 环境或移除 --require-gpu。",
                file=sys.stderr,
            )
            raise RuntimeError("GPU 不可用。")
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        print(f"已检测到 GPU：{device_name} (设备号：{device_idx})")
        return device_idx, f"cuda:{device_idx}"

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        print(
            f"检测到 GPU：{device_name} (设备号：{device_idx})，"
            "但未指定 --require-gpu，默认使用 CPU。"
        )
    return False, "cpu"


def _load_replay_buffer(path: str) -> ReplayBuffer:
    """把 .h5 读成 ReplayBuffer，统一带一点日志。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到离线数据集 {path}，请先运行 gen_datasets.py。")
    buffer = ReplayBuffer.load(path, buffer=InfiniteBuffer())
    print(f"成功加载 {buffer.size()} 条转移数据，来自 {path}")
    return buffer


def main() -> None:
    cli_args = _parse_args()
    custom_q_fn_tag = _maybe_patch_custom_factories()

    use_gpu_flag, device_str = _prepare_device(cli_args.require_gpu)

    is_fast = cli_args.fast_dev or EXPERIMENT_MODE in _FAST_MODE_NAMES
    target_steps = cli_args.steps
    if is_fast:
        target_steps = min(target_steps, FAST_BCQ_STEP_CAP)

    print("\n--- BCQ 实验管线：开始训练 DiscreteBCQ ---")
    print(f"训练步数: {target_steps} (mode={EXPERIMENT_MODE}, fast_dev={cli_args.fast_dev})")
    print(f"学习率: {cli_args.learning_rate}")
    print(f"批大小: {cli_args.batch_size}")
    print(f"奖励缩放: {cli_args.reward_scale}")
    print(f"设备: {device_str}")

    try:
        replay_buf = _load_replay_buffer(cli_args.dataset)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"错误：无法加载数据集 {cli_args.dataset}。{exc}")
        print("请确认数据集路径或先运行 hybrid_advisor_offline.offline.trainrl.gen_datasets。")
        return

    apply_reward_scale(replay_buf, cli_args.reward_scale)
    reward_scale_applied = getattr(replay_buf, "_reward_scale_applied", cli_args.reward_scale)

    bcq_cfg_kwargs = {
        "learning_rate": cli_args.learning_rate,
        "batch_size": cli_args.batch_size,
        "gamma": 0.99,
    }
    if custom_q_fn_tag:
        bcq_cfg_kwargs["q_func_factory"] = custom_q_fn_tag

    bcq_config = DiscreteBCQConfig(**bcq_cfg_kwargs)
    bcq_runner = bcq_config.create(device=use_gpu_flag)

    print("BCQ 算法已初始化，开始训练 ...")

    bcq_runner.fit(
        replay_buf,
        n_steps=target_steps,
        n_steps_per_epoch=min(target_steps, STEPS_PER_EPOCH),
        show_progress=True,
    )

    print("BCQ 训练完成，开始保存模型 ...")

    model_dir = os.path.dirname(cli_args.model_output)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    bcq_runner.save_model(cli_args.model_output)

    config = {
        "model_type": "DiscreteBCQ",
        "algo_name": "DiscreteBCQ",
        "steps": target_steps,
        "learning_rate": cli_args.learning_rate,
        "batch_size": cli_args.batch_size,
        "reward_scale": reward_scale_applied,
        "dataset_path": cli_args.dataset,
    }
    config_path = f"{cli_args.model_output}.config.json"
    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

    legacy_path = cli_args.model_output.replace(".pt", ".config.json")
    if legacy_path != config_path:
        legacy_dir = os.path.dirname(legacy_path)
        if legacy_dir:
            os.makedirs(legacy_dir, exist_ok=True)
        with open(legacy_path, "w", encoding="utf-8") as legacy_file:
            json.dump(config, legacy_file, ensure_ascii=False, indent=2)

    print(f"BCQ 模型已保存至：{cli_args.model_output}")
    print(f"配置文件已保存至：{config_path}")
    print(
        "下游评估（FQE/CPE）请使用同一份 reward_scale，"
        "并在 README 的 BCQ 示例中参考标准命令。"
    )


if __name__ == "__main__":
    main()
