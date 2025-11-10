"""
简化版 pipeline 脚本：按顺序把现有命令串起来，避免在同一进程里堆太多内存。

默认流程（stages=generate,train,eval）：
1. 生成离线数据集（可通过环境变量调整用户数、episode 长度等）。
2. 训练 BC / BCQ / CQL。
3. 逐个模型运行 eval_policy（单进程执行，避免 FQE 同时常驻）。

示例：
    python -m hybrid_advisor_offline.offline.run_full_pipeline \
      --dataset ./data/offline_dataset_small.h5 \
      --stages generate,train,eval \
      --use-gpu
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "offline_dataset.h5"
MODELS = {
    "bc": PROJECT_ROOT / "models" / "bc_model.pt",
    "bcq": PROJECT_ROOT / "models" / "bcq_discrete_model.pt",
    "cql": PROJECT_ROOT / "models" / "cql_discrete_model.pt",
}
TRAIN_MODULE = {
    "bc": "hybrid_advisor_offline.offline.trainrl.train_bc",
    "bcq": "hybrid_advisor_offline.offline.trainrl.train_bcq",
    "cql": "hybrid_advisor_offline.offline.trainrl.train_cql",
}


def _env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [p for p in existing.split(os.pathsep) if p]
    src = str(SRC_DIR)
    if src not in paths:
        paths.insert(0, src)
        env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run(label: str, cmd: List[str]) -> None:
    print(f"[pipeline] {label}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_env(), check=True)


def _behavior_meta_path(dataset: Path) -> Path:
    base = dataset.with_suffix("")
    return base.with_name(f"{base.name}_behavior.npz")


def _parse_stages(stage_arg: str) -> List[str]:
    if not stage_arg:
        return ["generate", "train", "eval"]
    stage_arg = stage_arg.strip().lower()
    if stage_arg in {"all", "default"}:
        return ["generate", "train", "eval"]
    return [s for s in (part.strip() for part in stage_arg.split(",")) if s]


def _maybe_add_gpu(cmd: List[str], use_gpu: bool) -> List[str]:
    if use_gpu:
        cmd = [*cmd, "--require-gpu"]
    return cmd


def _run_generation(dataset: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "hybrid_advisor_offline.offline.trainrl.gen_datasets",
        "--dataset-path",
        str(dataset),
    ]
    _run("generate", cmd)


def _run_training(dataset: Path, use_gpu: bool) -> None:
    for name, module in TRAIN_MODULE.items():
        model_path = MODELS[name]
        cmd = [
            sys.executable,
            "-m",
            module,
            "--dataset",
            str(dataset),
            "--model-output",
            str(model_path),
        ]
        cmd = _maybe_add_gpu(cmd, use_gpu)
        _run(f"train:{name}", cmd)


def _run_eval(
    dataset: Path,
    *,
    use_gpu: bool,
    fqe_steps: int | None,
    eval_interval: int | None,
    validation_ratio: float | None,
    fast_dev: bool,
) -> None:
    behavior_meta = _behavior_meta_path(dataset)
    for name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"[pipeline] 跳过 {name}，模型文件不存在：{model_path}")
            continue
        cmd = [
            sys.executable,
            "-m",
            "hybrid_advisor_offline.offline.eval.eval_policy",
            "--dataset",
            str(dataset),
            "--model",
            str(model_path),
        ]
        if behavior_meta.exists():
            cmd += ["--behavior-meta", str(behavior_meta)]
        if fast_dev:
            cmd.append("--fast-dev")
        else:
            if fqe_steps is not None:
                cmd += ["--fqe-steps", str(fqe_steps)]
            if eval_interval is not None:
                cmd += ["--eval-interval", str(eval_interval)]
            if validation_ratio is not None:
                cmd += ["--validation-ratio", str(validation_ratio)]
        cmd = _maybe_add_gpu(cmd, use_gpu)
        _run(f"eval:{name}", cmd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把数据生成、训练、评估命令串起来的轻量 pipeline。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="离线数据集路径（既是生成输出也是训练/评估输入）。",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="generate,train,eval",
        help="要执行的阶段，逗号分隔，可选 generate/train/eval，例如 --stages train,eval。",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="在训练与评估命令中追加 --require-gpu。",
    )
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help="在评估阶段启用 --fast-dev，减少 FQE 步数。",
    )
    parser.add_argument(
        "--fqe-steps",
        type=int,
        default=None,
        help="覆盖 eval_policy 的 --fqe-steps。",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="覆盖 eval_policy 的 --eval-interval。",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help="覆盖 eval_policy 的 --validation-ratio。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = _parse_stages(args.stages)
    dataset = args.dataset
    dataset.parent.mkdir(parents=True, exist_ok=True)

    if "generate" in stages:
        _run_generation(dataset)
    if "train" in stages:
        _run_training(dataset, use_gpu=args.use_gpu)
    if "eval" in stages:
        _run_eval(
            dataset,
            use_gpu=args.use_gpu,
            fqe_steps=args.fqe_steps,
            eval_interval=args.eval_interval,
            validation_ratio=args.validation_ratio,
            fast_dev=args.fast_eval,
        )
    print("[pipeline] done.")


if __name__ == "__main__":
    main()
