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
import os
from typing import Any, Dict

from hybrid_advisor_offline.offline.eval.fqe_data import (
    DATASET_PATH_DEFAULT,
    load_replay_buffer,
    prepare_fqe_datasets,
)
from hybrid_advisor_offline.offline.eval.cpe_metrics import compute_cpe_report
from hybrid_advisor_offline.offline.eval.fqe_runner import run_fqe
from hybrid_advisor_offline.offline.eval.policy_loader import (
    load_trained_policy,
    load_training_config,
)

try:
    from hybrid_advisor_offline.offline.cql.train_discrete import _require_gpu
except ImportError:  # pragma: no cover
    def _require_gpu() -> None:
        raise RuntimeError("GPU 校验函数缺失，请确认 train_discrete 模块可用。")

MODEL_PATH_DEFAULT = "./models/cql_discrete_model.pt"
BEHAVIOR_META_SUFFIX = "_behavior.npz"


def _behavior_meta_path(dataset_path: str, override: str | None) -> str | None:
    if override:
        return override
    base, ext = os.path.splitext(dataset_path)
    if not base:
        return None
    return f"{base}{BEHAVIOR_META_SUFFIX}"


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
    parser.add_argument(
        "--behavior-meta",
        type=str,
        default=None,
        help="行为策略倾向文件路径（默认与 dataset 同名前缀 + _behavior.npz）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.require_gpu:
        _require_gpu()

    train_cfg = load_training_config(args.model)
    reward_scale = train_cfg.get("reward_scale", 1.0)

    replay_buffer = load_replay_buffer(args.dataset, reward_scale=reward_scale)
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

    behavior_meta_path = _behavior_meta_path(args.dataset, args.behavior_meta)

    cpe_metrics: Dict[str, Any] = {}
    if args.no_cpe:
        print("跳过 CPE 评估。")
    else:
        print("--- 计算行为策略 CPE 指标 (IPS / SNIPS) ---")
        cpe_metrics = compute_cpe_report(
            replay_buffer,
            behavior_meta_path=behavior_meta_path,
        )

    summary = {
        "fqe": fqe_metrics,
        "cpe": cpe_metrics,
    }

    print("\n=== 评估结果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""
第一次评估问题：
=== 评估结果 === { "fqe": { "train_initial_state_value": 24.160295486450195, 
"train_average_value": 24.16844964198235, 
"val_initial_state_value": 20.5911808013916, 
"val_average_value": 20.592881008539315 }, 
"cpe": { "episode_return_mean": 1239.2344229354858, 
"ips": 0.23799393563508248, 
"snips": 0.23799393563508248 } }
问题：
FQE 输出的 val_average_value≈20 是折现价值（AverageValueEstimationEvaluator），单位接近“每步 reward/(1−γ)”。CPE 里的 episode_return_mean≈1239 则是 5 198 步累计回报，两者量纲不同，因此不能直接比较。
数据集中每步奖励约 0.238（市场收益 + 0.5×用户接受度），乘以 5 198 步正好得到 1 200+ 的 episode return，与 CPE 数字一致，说明评估流程本身没错，是解读出了偏差。
数据覆盖严重不足：规则策略几乎只访问动作 {0,2,4}，占 259 万步全部记录，动作 {1,3,5} 根本没有样本，CQL/FQE 对未见动作无从学习，自然难以超越基线。
训练和评估的奖励尺度不一致。CQL 用 StandardRewardScaler，FQE 则直接吃未经缩放的奖励，导致数值对不上；同时 50 000 步对 260 万步规模的 dataset 来说也远远不够。
IPS/SNIPS 因为没有行为策略概率（propensity），实际上退化成“平均每步奖励”，目前没有参考价值。

next step:
在 FQE 中新增“按 episode 累计回报”的 scorer，或把 FQE 的折现值换算成累计值，再与 CPE 的 episode_return_mean 比。
改进数据生成策略：为 rule_based_policy 加入随机扰动或多套脚本，并在轨迹里记录行为策略概率，方便后续真正计算 IPS/SNIPS/DR。
重新训练 CQL：提高 N_STEPS（至少与样本量同量级）、调大 conservative 系数等关键超参，并确保 FQE 使用与训练一致的奖励/观测 scaler；同时可考虑缩短 episode（例如每年重置）以减轻长序列带来的偏差。
"""

"""
第二次评估问题
=== 评估结果 ===
{
"fqe": {
"train_initial_state_value": 22.803264617919922,
"train_average_value": 22.80846461001513,
"val_initial_state_value": 22.754600524902344,
"val_average_value": 22.759357462554785,
"train_avg_reward_per_step": 0.2280846461001515,
"train_avg_episode_length": 5207.0,
"train_est_episode_return": 1187.636752243489,
"val_avg_reward_per_step": 0.22759357462554805,
"val_avg_episode_length": 5207.0,
"val_est_episode_return": 1185.0797430752286
},
"cpe": {
"episode_return_mean": 1189.7967247161864,
"ips": 0.22832407419602896,
"snips": 0.22832407419602896
}
}

FQE 现在直接把 AverageValueEstimationEvaluator 转成“每步奖励 + 估计的整段收益”，所以 train_avg_reward_per_step≈0.228、train_est_episode_return≈1188 跟 CPE 的 episode_return_mean≈1189.8 完全一致，说明评估并没有说策略更差，而是告诉我们：训练出的策略基本复制了行为策略。
造成这种“看起来还是很差”的原因主要有：
数据集仍由单一规则策略生成，ε 只有 0.1 且动作集只有 5 张卡，导致 RL 几乎只能拟合已有策略的行为；FQE/CPE 自然回到和 baseline 接近的数值。
奖励函数本身以“市场回报 + 0.5×接受概率”为主，均值≈0.228、总长≈5200 步，强化学习即使完全照抄也能得到同样的回报；但若想看到改进，需要更有区分度的奖励或缩短 episode（否则任何改进都被 5200 步平均化）。
训练配置还是默认的 N_STEPS=50k，远小于 260 万 transition；再加上 conservative loss 的 α=1.0，模型极度保守，不太可能探索新的分布。
"""
