"""
pipeline脚本，可整合整个离线强化学习工作流程：

1. （可选）下载市场和用户数据。
2. 生成包含个性化回撤奖励的离线数据集。
3. 训练所有现有算法（BC / BCQ / CQL）。
4. 对每个算法运行 FQE 和 CPE 评估。
5. 汇总每个细分市场的指标并生成对比图表。

用法：
python -m hybrid_advisor_offline.offline.run_full_pipeline \
--dataset ./data/offline_dataset.h5 \
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import joblib
import h5py

from hybrid_advisor_offline.offline.trainrl.gen_datasets import (
    download_mkt_data,
    download_user_data,
    generate_offline_dataset,
)
from hybrid_advisor_offline.offline.eval.fqe_data import (
    load_replay_buffer as load_eval_buffer,
    prepare_fqe_datasets,
)
from hybrid_advisor_offline.offline.eval.fqe_runner import run_fqe
from hybrid_advisor_offline.offline.eval.cpe_metrics import compute_cpe_report
from hybrid_advisor_offline.offline.eval.policy_loader import (
    load_training_config,
    load_trained_policy,
)
from hybrid_advisor_offline.offline.analysis.segment_metrics import (
    _load_replay_buffer as segment_load_buffer,
    _npz_to_profiles,
    aggregate_by_segment,
    collect_episode_snapshots,
    plot_segment_bars,
    EpisodeSnapshot,
    load_episode_summaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DATA_CSV = DATA_DIR / "mkt_data.csv"
USER_DATA_CSV = DATA_DIR / "bm_full.csv"
USER_MODEL_PATH = PROJECT_ROOT / "models" / "user_model.pkl"
EPISODE_SUMMARY_SUFFIX = "_episodes.csv"


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    label: str
    model_path: Path | None
    train_module: str | None = None
    train_args: tuple[str, ...] = ()
    supports_gpu: bool = True
    include_in_segments: bool = True
    value_from_policy: bool = True
    notes: Dict[str, str] = field(default_factory=dict)


ALGO_SPECS: tuple[AlgoSpec, ...] = (
    AlgoSpec(
        name="behavior",
        label="BehaviorPolicy",
        model_path=None,
        train_module=None,
        supports_gpu=False,
        include_in_segments=True,
        value_from_policy=False,
        notes={"description": "原始规则策略，直接使用 dataset 中记录的回报。"},
    ),
    AlgoSpec(
        name="bc",
        label="BehaviorCloning",
        model_path=PROJECT_ROOT / "models" / "bc_model.pt",
        train_module="hybrid_advisor_offline.offline.trainrl.train_bc",
        train_args=(),
        value_from_policy=False,
    ),
    AlgoSpec(
        name="bcq",
        label="DiscreteBCQ",
        model_path=PROJECT_ROOT / "models" / "bcq_discrete_model.pt",
        train_module="hybrid_advisor_offline.offline.trainrl.train_bcq",
        train_args=(),
    ),
    AlgoSpec(
        name="cql",
        label="DiscreteCQL",
        model_path=PROJECT_ROOT / "models" / "cql_discrete_model.pt",
        train_module="hybrid_advisor_offline.offline.trainrl.train_cql",
        train_args=(),
    ),
)


def _behavior_meta_path(dataset_path: Path) -> Path:
    base = dataset_path.with_suffix("")
    return base.with_name(f"{base.name}_behavior.npz")


def _episode_summary_path(dataset_path: Path) -> Path:
    base = dataset_path.with_suffix("")
    return base.with_name(f"{base.name}_episodes.csv")


def _python_env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    src_str = str(SRC_DIR)
    paths = existing.split(os.pathsep) if existing else []
    if src_str not in paths:
        paths.insert(0, src_str)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
    return env


def _run_cmd(cmd: List[str]) -> None:
    cmd_display = " ".join(cmd)
    print(f"[pipeline] >>> {cmd_display}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=_python_env(), check=True)


def _ensure_data(skip_download: bool, force_download: bool) -> None:
    if skip_download:
        print("[pipeline] 跳过数据下载阶段。")
        return
    if force_download or not MARKET_DATA_CSV.exists():
        print("[pipeline] 下载市场数据 ...")
        download_mkt_data()
    else:
        print(f"[pipeline] 市场数据已存在：{MARKET_DATA_CSV}")

    if force_download or not USER_DATA_CSV.exists():
        print("[pipeline] 下载用户画像数据 ...")
        download_user_data()
    else:
        print(f"[pipeline] 用户画像数据已存在：{USER_DATA_CSV}")


def _refresh_state_builder_cache(force: bool = False) -> None:
    """
    state_builder 在模块导入时缓存 user_model，如果导入时模型不存在，需要在生成数据前刷新。
    """
    try:
        from hybrid_advisor_offline.engine.state import state_builder
    except ImportError as exc:  # pragma: no cover - should not happen in normal runs
        raise RuntimeError("无法导入 state_builder，项目结构可能已变。") from exc

    needs_reload = force or getattr(state_builder, "_user_model_preprocessor", None) is None
    if not needs_reload:
        return

    if not USER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到用户模型 {USER_MODEL_PATH}，无法刷新 state_builder 缓存。"
        )

    pipeline = joblib.load(USER_MODEL_PATH)
    state_builder._user_model_pipeline = pipeline
    state_builder._user_model_preprocessor = pipeline.named_steps["preprocessor"]
    state_builder._state_dim = None
    print("[pipeline] 已刷新 state_builder 用户模型缓存。")


def _ensure_user_model(skip_train: bool, force_train: bool) -> None:
    if skip_train:
        print("[pipeline] 跳过用户模型训练。")
        return
    if USER_MODEL_PATH.exists() and not force_train:
        print(f"[pipeline] 用户模型已存在：{USER_MODEL_PATH}")
        return

    print(f"[pipeline] 训练用户模型 -> {USER_MODEL_PATH}")
    cmd = [
        sys.executable,
        "-m",
        "hybrid_advisor_offline.offline.trainrl.train_usr_model",
        "--data",
        str(USER_DATA_CSV),
        "--output",
        str(USER_MODEL_PATH),
    ]
    _run_cmd(cmd)


def _ensure_dataset(dataset_path: Path, force_regen: bool) -> None:
    if dataset_path.exists() and not force_regen:
        print(f"[pipeline] 离线数据集已存在：{dataset_path}")
        return
    print(f"[pipeline] 生成离线数据集 -> {dataset_path}")
    os.environ["OUTPUT_DATASET_PATH"] = str(dataset_path)
    generate_offline_dataset()
    if not dataset_path.exists():
        raise RuntimeError(
            f"数据集生成失败：未找到 {dataset_path}。请检查前面的日志以定位问题。"
        )


def _train_algorithms(
    dataset_path: Path,
    require_gpu: bool,
    skip_train: bool,
) -> None:
    if skip_train:
        print("[pipeline] 跳过训练阶段。")
        return

    for spec in ALGO_SPECS:
        if spec.train_module is None or spec.model_path is None:
            continue
        cmd = [
            sys.executable,
            "-m",
            spec.train_module,
            "--dataset",
            str(dataset_path),
            "--model-output",
            str(spec.model_path),
        ]
        cmd.extend(spec.train_args)
        if require_gpu and spec.supports_gpu:
            cmd.append("--require-gpu")
        try:
            _run_cmd(cmd)
        except subprocess.CalledProcessError as exc:
            print(f"[pipeline] 训练 {spec.label} 失败：{exc}")
            raise


def _run_policy_eval(
    algo_name: str,
    model_path: Path,
    dataset_path: Path,
    behavior_meta_path: Path,
    *,
    fqe_steps: int,
    eval_interval: int,
    validation_ratio: float,
    require_gpu: bool,
    log_dir: Path,
) -> Dict[str, Dict]:
    cfg = load_training_config(str(model_path))
    reward_scale = cfg.get("reward_scale", 1.0)

    buffer = load_eval_buffer(str(dataset_path), reward_scale=reward_scale)
    policy = load_trained_policy(
        str(model_path),
        buffer,
        require_gpu=require_gpu,
    )

    train_dataset, val_dataset = prepare_fqe_datasets(
        buffer,
        validation_ratio=validation_ratio,
    )
    algo_log_dir = log_dir / algo_name
    algo_log_dir.mkdir(parents=True, exist_ok=True)
    fqe_metrics = run_fqe(
        policy,
        train_dataset,
        val_dataset,
        n_steps=fqe_steps,
        eval_interval=eval_interval,
        log_dir=str(algo_log_dir),
        require_gpu=require_gpu,
    )
    cpe_metrics = compute_cpe_report(
        buffer,
        behavior_meta_path=str(behavior_meta_path),
    )
    return {"fqe": fqe_metrics, "cpe": cpe_metrics}


def _evaluate_algorithms(
    dataset_path: Path,
    behavior_meta_path: Path,
    output_dir: Path,
    *,
    skip_eval: bool,
    fqe_steps: int,
    eval_interval: int,
    validation_ratio: float,
    require_gpu: bool,
) -> Dict[str, Dict]:
    if skip_eval:
        print("[pipeline] 跳过评估阶段。")
        return {}

    summary: Dict[str, Dict] = {}
    eval_dir = output_dir / "policy_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_dir = eval_dir / "fqe_logs"

    for spec in ALGO_SPECS:
        if spec.model_path is None:
            continue
        if not spec.model_path.exists():
            print(f"[pipeline] 警告：{spec.label} 模型缺失，跳过评估。")
            continue
        print(f"[pipeline] 评估 {spec.label} ...")
        metrics = _run_policy_eval(
            spec.name,
            spec.model_path,
            dataset_path,
            behavior_meta_path,
            fqe_steps=fqe_steps,
            eval_interval=eval_interval,
            validation_ratio=validation_ratio,
            require_gpu=require_gpu,
            log_dir=log_dir,
        )
        summary[spec.name] = metrics
        with open(eval_dir / f"{spec.name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    if summary:
        with open(eval_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _segment_visuals(
    dataset_path: Path,
    behavior_meta_path: Path,
    episode_summary_path: Path | None,
    output_dir: Path,
    *,
    skip_plots: bool,
) -> None:
    if skip_plots:
        print("[pipeline] 跳过分群图表生成。")
        return
    if not behavior_meta_path.exists():
        raise FileNotFoundError(
            f"找不到行为策略 meta：{behavior_meta_path}，请重新生成数据集。"
        )
    profiles = _npz_to_profiles(str(behavior_meta_path))
    combined: List[pd.DataFrame] = []
    shared_buffer = None
    streaming_snapshots: List = []
    streaming_loaded = False
    summary_snapshots = load_episode_summaries(str(episode_summary_path)) if episode_summary_path else None

    def _resolve_buffer(use_policy: bool):
        nonlocal shared_buffer
        if use_policy:
            return segment_load_buffer(str(dataset_path))
        if shared_buffer is None:
            shared_buffer = segment_load_buffer(str(dataset_path))
        return shared_buffer

    def _load_streaming_snapshots():
        nonlocal streaming_snapshots, streaming_loaded
        if streaming_loaded:
            return streaming_snapshots
        snapshots: List = []
        with h5py.File(str(dataset_path), "r") as h5_file:
            reward_keys = [
                key for key in h5_file.keys()
                if key.startswith("rewards_")
            ]
            reward_keys.sort(key=lambda k: int(k.split("_")[1]))
            for idx, key in enumerate(reward_keys):
                rewards = np.asarray(h5_file[key], dtype=np.float64).reshape(-1)
                total_reward = float(rewards.sum())
                length = int(rewards.size)
                snapshots.append(EpisodeSnapshot(idx, total_reward, length))
        streaming_snapshots = snapshots
        streaming_loaded = True
        return streaming_snapshots

    for spec in ALGO_SPECS:
        if not spec.include_in_segments:
            continue
        if spec.value_from_policy:
            buffer = _resolve_buffer(True)
            policy = None
            if spec.model_path is not None:
                if not spec.model_path.exists():
                    print(f"[pipeline] 警告：缺少 {spec.label} 模型，跳过 segment 统计。")
                    continue
                policy = load_trained_policy(
                    str(spec.model_path),
                    buffer,
                    require_gpu=False,
                )
            snapshots = collect_episode_snapshots(buffer, policy=policy)
        else:
            if summary_snapshots is not None:
                snapshots = summary_snapshots
            else:
                snapshots = _load_streaming_snapshots()
        stats = aggregate_by_segment(spec.label, snapshots, profiles)
        combined.append(stats)

    if not combined:
        print("[pipeline] 未找到可用的 segment 统计数据。")
        return

    merged = pd.concat(combined, ignore_index=True)
    vis_dir = output_dir / "segment_reports"
    vis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = vis_dir / "segment_metrics.csv"
    fig_path = vis_dir / "segment_metrics.png"
    merged.to_csv(csv_path, index=False)
    plot_segment_bars(merged, fig_path)
    print(f"[pipeline] segment CSV: {csv_path}")
    print(f"[pipeline] segment FIG: {fig_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键跑通离线 RL 全流程（生成数据 -> 训练 -> 评估 -> 图表）。",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "offline_dataset.h5",
        help="离线数据集输出/输入路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "full_pipeline",
        help="全流程产物输出目录（评估报告、图表等）。",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="跳过市场/用户数据下载阶段。",
    )
    parser.add_argument(
        "--skip-user-model",
        action="store_true",
        help="跳过用户接受度模型训练（需确保 models/user_model.pkl 已存在）。",
    )
    parser.add_argument(
        "--force-user-model",
        action="store_true",
        help="无论现有文件与否都重新训练用户模型。",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载市场与用户数据。",
    )
    parser.add_argument(
        "--force-regen",
        action="store_true",
        help="即使离线数据集已存在也重新生成。",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过算法训练阶段。",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过 FQE/CPE 评估阶段。",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="跳过分群图表生成。",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="训练和 FQE 强制使用 GPU。",
    )
    parser.add_argument(
        "--fqe-steps",
        type=int,
        default=100_000,
        help="FQE 评估迭代步数。",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10_000,
        help="FQE 评估间隔。",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="FQE 训练/验证划分比例。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.chdir(PROJECT_ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    behavior_meta = _behavior_meta_path(args.dataset)
    episode_summary = _episode_summary_path(args.dataset)

    _ensure_data(skip_download=args.skip_download, force_download=args.force_download)
    _ensure_user_model(skip_train=args.skip_user_model, force_train=args.force_user_model)
    _refresh_state_builder_cache(force=args.force_user_model or not args.skip_user_model)
    _ensure_dataset(args.dataset, force_regen=args.force_regen)
    _train_algorithms(
        args.dataset,
        require_gpu=args.require_gpu,
        skip_train=args.skip_train,
    )
    _evaluate_algorithms(
        args.dataset,
        behavior_meta,
        args.output_dir,
        skip_eval=args.skip_eval,
        fqe_steps=args.fqe_steps,
        eval_interval=args.eval_interval,
        validation_ratio=args.validation_ratio,
        require_gpu=args.require_gpu,
    )
    _segment_visuals(
        args.dataset,
        behavior_meta,
        episode_summary,
        args.output_dir,
        skip_plots=args.skip_plots,
    )
    print("[pipeline] 全流程执行完毕。")


if __name__ == "__main__":
    main()

# WSL 被杀掉是因为 segment 阶段要重新把 data/offline_dataset.h5 整个读成 d3rlpy 的 ReplayBuffer。
# 这个 .h5 只有 4.1 GB，但 ReplayBuffer.load 会把所有 episode 的 observation/action/reward 全部解包进 Python 对象，
# 内存瞬间冲到 11 GB 以上。你这次运行的命令把前面的所有步骤都 skip 掉，只剩下 _segment_visuals，所以一启动就加载整套数据，WSL 直接 OOM
# [  637.539319] [    826]     0   826     1705      576    53248      128             0 login
# [  637.539330] [    868]  1000   868     5082      704    86016      128           100 systemd
# [  637.539331] [    869]  1000   869     5288      623    77824      192           100 (sd-pam)
# [  637.539341] [    900]  1000   900     2974      640    65536       64             0 bash
# [  637.539343] [   2446]  1000  2446 16767719  2849942 42250240  2083101             0 python
# [  637.539492] oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=/,mems_allowed=0,global_oom,task_memcg=/init.scope,task=python,pid=2446,uid=1000
# [  637.543623] Out of memory: Killed process 2446 (python) total-vm:67070876kB, anon-rss:11399768kB, file-rss:0kB, shmem-rss:0kB, UID:1000 pgtables:41260kB oom_score_adj:0
