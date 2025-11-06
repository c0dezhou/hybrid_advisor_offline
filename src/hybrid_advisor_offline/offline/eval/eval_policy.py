# FQE（Fitted Q Evaluation）是一种离线策略评估方法：先固定目标策略（训练好的 CQL 策略），
# 然后用离线数据迭代训练一个 Q 函数逼近器，仅用于评估，不参与策略优化；
# 最后用这个 Q 函数估计策略的长期回报。

# 评估入口：FQE + CPE + 环境回测 + 公平性检查。
# 目前没做回测和公平性
# python -m hybrid_advisor_offline.offline.eval.eval_policy \
#   --dataset ./data/offline_dataset.h5 \
#   --model ./models/cql_discrete_model.pt

import argparse
import json
from typing import Any, Dict

from hybrid_advisor_offline.offline.eval.fqe_data import (
    DATASET_PATH_DEFAULT,
    load_replay_buffer,
    prepare_fqe_datasets,
)
from hybrid_advisor_offline.offline.eval.cpe_metrics import compute_cpe_report
from hybrid_advisor_offline.offline.eval.fqe_runner import run_fqe
from hybrid_advisor_offline.offline.eval.policy_loader import load_trained_policy

try:
    from hybrid_advisor_offline.offline.cql.train_discrete import _require_gpu
except ImportError:  # pragma: no cover
    def _require_gpu() -> None:
        raise RuntimeError("GPU 校验函数缺失，请确认 train_discrete 模块可用。")

MODEL_PATH_DEFAULT = "./models/cql_discrete_model.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 FQE + 回测 + 公平性检查对离线策略做综合评估。",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_PATH_DEFAULT,
        help="离线数据集路径 (默认: ./data/offline_dataset.h5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH_DEFAULT,
        help="训练好的离散 CQL 模型路径 (默认: ./models/cql_discrete_model.pt)",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="需要在 GPU 上运行策略和 FQE 时使用该参数。",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="FQE 验证集划分比例 (默认: 0.1)。",
    )
    parser.add_argument(
        "--fqe-steps",
        type=int,
        default=100_000,
        help="FQE 迭代步数 (默认: 100000)。",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10_000,
        help="FQE 评估间隔步数 (默认: 10000)。",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="d3rlpy_logs/fqe",
        help="FQE 日志输出目录。",
    )
    parser.add_argument(
        "--no-cpe",
        action="store_true",
        help="跳过基于行为策略的 CPE 指标。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.require_gpu:
        _require_gpu()

    replay_buffer = load_replay_buffer(args.dataset)
    policy = load_trained_policy(
        args.model,
        replay_buffer,
        require_gpu=args.require_gpu,
    )

    train_dataset, val_dataset = prepare_fqe_datasets(
        replay_buffer,
        validation_ratio=args.validation_ratio,
    )

    print("--- 运行 Fitted Q Evaluation (FQE) ---")
    fqe_metrics = run_fqe(
        policy,
        train_dataset,
        val_dataset,
        n_steps=args.fqe_steps,
        eval_interval=args.eval_interval,
        log_dir=args.log_dir,
        require_gpu=args.require_gpu,
    )

    cpe_metrics: Dict[str, Any] = {}
    if args.no_cpe:
        print("跳过 CPE 评估。")
    else:
        print("--- 计算行为策略 CPE 指标 (IPS / SNIPS) ---")
        cpe_metrics = compute_cpe_report(replay_buffer)

    summary = {
        "fqe": fqe_metrics,
        "cpe": cpe_metrics,
    }

    print("\n=== 评估结果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
