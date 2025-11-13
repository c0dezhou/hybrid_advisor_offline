"""训练一个行为克隆（BC）基线模型来诊断状态的有效性。
BC模型忽略奖励，纯粹模仿行为策略。通过将其评估结果与历史表现进行比较，可以判断
91维观测向量是否包含足够的学习信息。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import torch
from d3rlpy.algos import DiscreteBCConfig
from d3rlpy.dataset import ReplayBuffer
from d3rlpy.dataset.buffers import InfiniteBuffer

from hybrid_advisor_offline.offline.utils.reward_scaling import apply_reward_scale

DEFAULT_DATASET = "./data/offline_dataset_train.h5"
DEFAULT_MODEL_OUTPUT = "./models/bc_model.pt"
DEFAULT_STEPS = 500_000
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 256
DEFAULT_REWARD_SCALE = float(os.getenv("BC_REWARD_SCALE", "1000.0"))
STEPS_PER_EPOCH = int(os.getenv("BC_STEPS_PER_EPOCH", "5000"))
EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "full").lower()
_FAST_MODE_NAMES = {"fast", "dev"}
FAST_BC_STEP_CAP = int(os.getenv("BC_FAST_STEP_CAP", "100000"))


def _maybe_patch_custom_factories() -> None:
    """Optionally override d3rlpy network factory if the repo provides it."""

    if os.getenv("USE_CARD_FACTORY", "0") != "1":
        return

    try:
        from hybrid_advisor_offline.offline.utils.network_factories import FCHW
        import d3rlpy.models.q_functions as q_functions
    except ImportError:
        print("警告：USE_CARD_FACTORY=1，但未找到自定义 FCHW，跳过自定义网络。")
        return

    q_functions.FCHW = FCHW
    print("已启用自定义 FCHW 网络工厂。")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V2.2.19 诊断脚本：训练行为克隆 (BC) 基线。",
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
        help=f"BC 模型权重输出路径（默认 {DEFAULT_MODEL_OUTPUT}）。",
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
        help="快速实验模式：自动缩短训练步数，方便调参。",
    )
    return parser.parse_args()


def _prepare_device(require_gpu: bool) -> Tuple[int | bool, str]:
    """Return the d3rlpy use_gpu flag and a readable device string."""

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
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到离线数据集 {path}，请先运行 gen_datasets.py。")
    buffer = ReplayBuffer.load(path, buffer=InfiniteBuffer())
    print(f"成功加载 {buffer.size()} 条转移数据，来自 {path}")
    return buffer


def main() -> None:
    args = _parse_args()
    _maybe_patch_custom_factories()

    use_gpu_flag, device_str = _prepare_device(args.require_gpu)

    is_fast = args.fast_dev or EXPERIMENT_MODE in _FAST_MODE_NAMES
    target_steps = args.steps
    if is_fast:
        target_steps = min(target_steps, FAST_BC_STEP_CAP)

    print("\n--- V2.2.19 诊断：开始训练行为克隆 (BC) 基线 ---")
    print(f"训练步数: {target_steps} (mode={EXPERIMENT_MODE}, fast_dev={args.fast_dev})")
    print(f"学习率: {args.learning_rate}")
    print(f"批大小: {args.batch_size}")
    print(f"奖励缩放: {args.reward_scale}")
    print(f"设备: {device_str}")

    try:
        dataset = _load_replay_buffer(args.dataset)
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"错误：无法加载数据集 {args.dataset}。{exc}")
        print("请确认数据集路径或先运行 hybrid_advisor_offline.offline.trainrl.gen_datasets。")
        return

    apply_reward_scale(dataset, args.reward_scale)

    bc_config = DiscreteBCConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    bc = bc_config.create(device=use_gpu_flag)

    print("BC 算法已初始化，开始训练 ...")

    bc.fit(
        dataset,
        n_steps=target_steps,
        n_steps_per_epoch=min(target_steps, STEPS_PER_EPOCH),
        show_progress=True,
    )

    print("BC 训练完成，开始保存模型 ...")

    model_dir = os.path.dirname(args.model_output)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    bc.save_model(args.model_output)

    config = {
        "model_type": "BC",
        "steps": target_steps,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "reward_scale": args.reward_scale,
        "dataset_path": args.dataset,
    }
    preferred_config_path = f"{args.model_output}.config.json"
    legacy_config_path = args.model_output.replace(".pt", ".config.json")
    config_path = preferred_config_path
    config_dir = os.path.dirname(config_path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)
    if legacy_config_path != preferred_config_path:
        legacy_dir = os.path.dirname(legacy_config_path)
        if legacy_dir:
            os.makedirs(legacy_dir, exist_ok=True)
        with open(legacy_config_path, "w", encoding="utf-8") as legacy_file:
            json.dump(config, legacy_file, ensure_ascii=False, indent=2)

    print(f"BC 模型已保存至：{args.model_output}")
    print(f"配置文件已保存至：{config_path}")


if __name__ == "__main__":
    main()
