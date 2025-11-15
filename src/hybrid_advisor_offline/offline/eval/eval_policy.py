# FQE（Fitted Q Evaluation）是一种离线策略评估方法：先固定目标策略（训练好的 CQL 策略），
# 然后用离线数据迭代训练一个 Q 函数逼近器，仅用于评估，不参与策略优化；
# 最后用这个 Q 函数估计策略的长期回报。

# 评估入口：FQE + CPE + 环境回测 + 公平性检查。
# 加回测和公平性检查
# python -m hybrid_advisor_offline.offline.eval.eval_policy \
#   --dataset ./data/offline_dataset.h5 \
#   --model ./models/cql_discrete_model.pt

import argparse
import json
import os
import random
from typing import Any, Dict, Optional

import inspect
import numpy as np
import pandas as pd
import torch

from hybrid_advisor_offline.offline.eval.fqe_data import (
    DATASET_PATH_DEFAULT,
    load_replay_buffer,
    prepare_fqe_datasets,
)
from hybrid_advisor_offline.offline.eval.cpe_metrics import compute_cpe_report
from hybrid_advisor_offline.offline.eval.fqe_runner import run_fqe
from hybrid_advisor_offline.engine.act_safety.act_cards_factory import build_card_factory
from hybrid_advisor_offline.engine.envs.market_envs import MarketEnv
from hybrid_advisor_offline.engine.state.state_builder import (
    UserProfile,
    build_state_vec,
    make_up_to_vec,
)
from hybrid_advisor_offline.offline.eval.policy_loader import (
    load_trained_policy,
    load_training_config,
)

try:
    from hybrid_advisor_offline.offline.trainrl.train_cql import _require_gpu
except ImportError:  # pragma: no cover
    def _require_gpu() -> None:
        raise RuntimeError("GPU 校验函数缺失，请确认 train_discrete 模块可用。")

MODEL_PATH_DEFAULT = "./models/cql_discrete_model.pt"
BEHAVIOR_META_SUFFIX = "_behavior.npz"
EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "full").lower()
_FAST_MODE_NAMES = {"fast", "dev"}
_PROFILE_FIELDS = set(UserProfile.__dataclass_fields__)


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _observation_scaler_name(policy) -> str:
    impl = getattr(policy, "impl", policy)
    scaler = getattr(impl, "observation_scaler", None)
    if scaler is None:
        return "none"
    return scaler.__class__.__name__


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
    parser.add_argument(
        "--policy",
        choices=["trained", "behavior"],
        default="trained",
        help="指定执行哪类策略：有训练模型时选择 trained。",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="对策略做环境回测并输出 backtest_metrics。",
    )
    parser.add_argument(
        "--backtest-episodes",
        type=int,
        default=None,
        help="最多回测多少条验证集 episode；默认全量。",
    )
    parser.add_argument(
        "--backtest-seed",
        type=int,
        default=None,
        help="回测用的随机种子（默认取 VAL_START_SEED/0）。",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="快速实验模式：缩短 FQE 步数和评估间隔。",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="若提供路径，则把最终评估 JSON 另存为文件，方便交付。",
    )
    return parser.parse_args()


def _infer_risk_bucket(profile_dict: Dict[str, Any]) -> int | None:
    """复用 UserProfile.risk_bucket 的年龄分档逻辑。"""
    age = profile_dict.get("age")
    try:
        age = int(age)
    except (TypeError, ValueError):
        return None
    if age > 55:
        return 0
    if age > 35:
        return 1
    return 2
def _load_behavior_profiles(behavior_meta_path: Optional[str]) -> Optional[list[dict[str, Any]]]:
    if not behavior_meta_path or not os.path.exists(behavior_meta_path):
        return None
    try:
        meta = np.load(behavior_meta_path, allow_pickle=True)
    except OSError:
        return None
    meta_profiles = meta.get("user_profiles")
    if meta_profiles is None:
        return None
    profiles: list[dict[str, Any]] = []
    for item in meta_profiles:
        if isinstance(item, dict):
            profiles.append(item)
            continue
        try:
            obj = item.item()
            if isinstance(obj, dict):
                profiles.append(obj)
                continue
        except Exception:
            pass
        try:
            profiles.append(dict(item))
        except Exception:
            profiles.append({})
    return profiles


def _build_episode_bucket_map(
    profiles: Optional[list[dict[str, Any]]],
    num_episodes: int,
) -> Dict[int, int]:
    if not profiles:
        return {}
    buckets: Dict[int, int] = {}
    limit = min(len(profiles), num_episodes)
    for idx in range(limit):
        bucket = _infer_risk_bucket(profiles[idx])
        if bucket is not None:
            buckets[idx] = int(bucket)
    return buckets


def _collect_episode_metrics(
    replay_buffer,
    episode_bucket_map: Optional[Dict[int, int]] = None,
    *,
    reward_scale_fallback: float = 1.0,
) -> pd.DataFrame:
    episodes = list(replay_buffer.episodes)
    if not episodes:
        return pd.DataFrame(
            columns=["episode_id", "total_return", "max_drawdown", "risk_bucket"]
        )

    reward_scale_applied = float(
        getattr(replay_buffer, "_reward_scale_applied", reward_scale_fallback) or 1.0
    )
    inv_scale = 1.0 / reward_scale_applied if reward_scale_applied != 1.0 else 1.0

    rows: list[dict[str, Any]] = []
    for ep_id, episode in enumerate(episodes):
        step_rewards = np.asarray(episode.rewards, dtype=np.float64).reshape(-1)
        if inv_scale != 1.0:
            step_rewards = step_rewards * inv_scale
        total_ret = float(step_rewards.sum())
        if step_rewards.size:
            cum_rewards = np.cumsum(step_rewards)
            worst_cum = float(cum_rewards.min())
        else:
            worst_cum = 0.0
        max_dd = worst_cum
        risk_bucket = episode_bucket_map.get(ep_id) if episode_bucket_map else None
        rows.append(
            {
                "episode_id": ep_id,
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "risk_bucket": risk_bucket,
            }
        )
    return pd.DataFrame(rows)


def _build_risk_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"overall": {"n_episodes": 0}}

    def _agg(frame: pd.DataFrame) -> Dict[str, float]:
        return {
            "n_episodes": int(len(frame)),
            "avg_total_return": float(frame["total_return"].mean()),
            "avg_max_drawdown": float(frame["max_drawdown"].mean()),
        }

    metrics: Dict[str, Any] = {"overall": _agg(df)}

    if "risk_bucket" in df.columns:
        by_bucket: Dict[str, Any] = {}
        for bucket, group in df.dropna(subset=["risk_bucket"]).groupby("risk_bucket"):
            by_bucket[str(int(bucket))] = _agg(group)
        if by_bucket:
            metrics["by_risk_bucket"] = by_bucket
    return metrics


def _compute_fqe_by_bucket(
    fqe_model,
    replay_buffer,
    episode_bucket_map: Optional[Dict[int, int]],
) -> Dict[str, Any]:
    if fqe_model is None or not episode_bucket_map:
        return {}
    bucket_values: Dict[int, list[float]] = {}
    for ep_id, episode in enumerate(replay_buffer.episodes):
        bucket = episode_bucket_map.get(ep_id)
        if bucket is None:
            continue
        observations = np.asarray(episode.observations, dtype=np.float32)
        if observations.size == 0:
            continue
        state = observations[:1]
        try:
            value = float(fqe_model.predict_value(state)[0])
        except Exception:
            continue
        bucket_values.setdefault(bucket, []).append(value)
    stats: Dict[str, Any] = {}
    for bucket, values in bucket_values.items():
        if not values:
            continue
        stats[str(bucket)] = {
            "n_episodes": len(values),
            "avg_value": float(np.mean(values)),
        }
    return stats


def _dict_to_user_profile(profile_data: Any) -> Optional[UserProfile]:
    if profile_data is None:
        return None
    if isinstance(profile_data, dict):
        source = profile_data
    else:
        try:
            source = profile_data.item()
        except Exception:
            try:
                source = dict(profile_data)
            except Exception:
                return None
    filtered = {k: source.get(k) for k in _PROFILE_FIELDS if k in source}
    if not filtered:
        return None
    try:
        return UserProfile(**filtered)
    except TypeError:
        return None


def _call_policy_predict(
    policy,
    obs: np.ndarray,
    *,
    n_actions: int | None = None,
) -> int:
    fn = getattr(policy, "predict", None)
    if fn is None:
        for name in ("predict_action", "sample_action"):
            if hasattr(policy, name):
                fn = getattr(policy, name)
                break
    if fn is None or not callable(fn):
        raise TypeError("Policy has no predict/predict_action/sample_action method.")

    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(fn)
        if "deterministic" in sig.parameters:
            kwargs["deterministic"] = True
    except (TypeError, ValueError):
        pass

    out = fn(obs, **kwargs) if kwargs else fn(obs)

    try:
        import torch

        if hasattr(out, "detach") and torch.is_tensor(out):
            out = out.detach().cpu().numpy()
    except Exception:
        pass

    if isinstance(out, (list, tuple)):
        out = out[0]
    arr = np.asarray(out)

    if arr.dtype.kind in ("i", "u"):
        return int(np.squeeze(arr))

    if arr.ndim == 2 and arr.shape[0] == 1 and (
        n_actions is None or arr.shape[1] == n_actions
    ):
        return int(np.argmax(arr[0]))

    if arr.ndim == 1 and (n_actions is None or arr.shape[0] == n_actions):
        return int(np.argmax(arr))

    return int(np.squeeze(arr))


def _rollout_policy(
    policy,
    env,
    steps: int,
    user_profile: UserProfile,
    *,
    n_actions: int,
):
    info = env.reset()
    if user_profile is None:
        raise ValueError("无法从行为元数据构建 user profile，回测终止。")
    user_vector = make_up_to_vec(user_profile)
    rets: list[float] = []
    import torch

    with torch.no_grad():
        for _ in range(steps):
            state_vec = build_state_vec(
                mkt_features=info["mkt_sshot"],
                user_profile=user_profile,
                curr_alloc=info["curr_alloc"],
                user_vector=user_vector,
            )
            obs = np.asarray(state_vec, dtype=np.float32).reshape(1, -1)
            act = _call_policy_predict(policy, obs, n_actions=n_actions)
            if not (0 <= act < n_actions):
                raise ValueError(
                    f"Predicted action {act} out of range [0, {n_actions - 1}]"
                )
            market_return, terminated, next_info = env.step(int(act))
            rets.append(float(next_info.get("market_return", market_return)))
            info = next_info
            if terminated:
                break
    return np.asarray(rets, np.float64)


def _series_to_metrics(step_returns: np.ndarray):
    if step_returns.size == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "cvar5": 0.0,
            "mean_step_return": 0.0,
            "n_steps": 0,
        }
    equity = np.cumprod(np.clip(1.0 + step_returns, 1e-6, None))
    peak = np.maximum.accumulate(equity)
    dd = equity / np.clip(peak, 1e-6, None) - 1.0
    total = float(equity[-1] - 1.0)
    max_dd = float(dd.min())
    mu = float(step_returns.mean())
    sd = float(step_returns.std(ddof=1))
    vol = sd * np.sqrt(252.0)
    if sd > 1e-8:
        sharpe = float(mu / sd * np.sqrt(252.0))
    else:
        sharpe = 0.0
    q = np.quantile(step_returns, 0.05)
    tail = step_returns[step_returns <= q]
    cvar5 = float(tail.mean()) if tail.size else float(q)
    return {
        "total_return": total,
        "max_drawdown": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "cvar5": cvar5,
        "mean_step_return": mu,
        "n_steps": int(step_returns.size),
    }


def run_backtest(
    policy,
    behavior_meta_path: Optional[str],
    dataset_path: str,
    policy_name: str,
    action_size: Optional[int] = None,
    max_episodes: Optional[int] = None,
    behavior_profiles: Optional[list[dict[str, Any]]] = None,
    seed: Optional[int] = None,
):
    if policy is None:
        return {}
    if not behavior_meta_path or not os.path.exists(behavior_meta_path):
        raise FileNotFoundError("回测必须提供验证集行为 meta 文件。")

    card_ids = sorted(card.act_id for card in build_card_factory())
    assert card_ids == list(range(len(card_ids))), "ActCard.act_id 未覆盖连续编号。"
    resolved_action_size = action_size if action_size is not None else len(card_ids)
    if resolved_action_size != len(card_ids):
        raise AssertionError(
            "模型 action_size 与卡片数量不一致，无法回测。"
        )
    n_actions = int(resolved_action_size)

    actual_seed = int(
        seed
        if seed is not None
        else os.environ.get("BACKTEST_SEED", os.environ.get("VAL_START_SEED", "0"))
    )
    _set_global_seeds(actual_seed)
    policy_impl = getattr(policy, "impl", None)
    if hasattr(policy, "eval"):
        policy.eval()
    elif policy_impl is not None and hasattr(policy_impl, "eval"):
        policy_impl.eval()

    if behavior_profiles is None:
        behavior_profiles = _load_behavior_profiles(behavior_meta_path) or []

    try:
        with np.load(behavior_meta_path, allow_pickle=True) as meta:
            starts = meta["episode_start_idx"].astype(int)
            steps_arr = meta.get("episode_steps")
            steps = int(steps_arr[0]) if steps_arr is not None and len(steps_arr) else 252
            splits = meta.get("episode_split")
    except Exception as exc:  # pragma: no cover
        print("[eval_policy] 回测读取行为 meta 失败：", exc)
        return {}

    if starts.size == 0:
        raise RuntimeError("行为 meta 中没有 episode_start_idx，无法回测。")

    if splits is not None:
        uniq = np.unique(splits.astype(str))
        if len(uniq) == 1 and uniq[0].lower() != "val":
            raise RuntimeError("回测必须使用验证集窗口，如 meta 中 split ≠ val。")

    if max_episodes is not None:
        starts = starts[:max_episodes]
        behavior_profiles = behavior_profiles[:max_episodes]

    profile_objs: list[Optional[UserProfile]] = [
        _dict_to_user_profile(profile) for profile in behavior_profiles
    ]

    metadata_module = policy_impl if policy_impl is not None else policy
    device = getattr(metadata_module, "device", "cpu")
    scaler_name = _observation_scaler_name(policy)
    print(
        "[eval_policy] 验证集回测",
        f"dataset={dataset_path}",
        f"episodes_to_consider={len(starts)}",
        f"seed={actual_seed}",
        f"device={device}",
        f"obs_scaler={scaler_name}",
    )

    base_env_vars = {
        "EPISODE_START_MODE": "fixed",
        "EPISODE_FIXED_START": "0",
        "EPISODE_START_SEED": "0",
        "EPISODE_START_POOL": "all",
    }
    old_env = {key: os.environ.get(key) for key in base_env_vars}
    os.environ.update(base_env_vars)
    try:
        env = MarketEnv(max_episode_steps=steps)
    finally:
        for key, old in old_env.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    env._start_mode = "fixed"

    rows: list[dict[str, Any]] = []
    for ep_id, start_idx in enumerate(starts):
        if ep_id >= len(profile_objs) or profile_objs[ep_id] is None:
            print(
                f"[eval_policy] 回测跳过 episode {ep_id}：缺少合法 user_profile。"
            )
            continue
        env._fixed_start_idx = int(start_idx)
        step_rets = _rollout_policy(
            policy,
            env,
            steps,
            profile_objs[ep_id],
            n_actions=n_actions,
        )
        metrics = _series_to_metrics(step_rets)
        risk_bucket = getattr(profile_objs[ep_id], "risk_bucket", None)
        rows.append(
            {
                "episode_id": ep_id,
                "episode_start_idx": int(start_idx),
                "policy": policy_name,
                "split": "val",
                "seed": actual_seed,
                "age": getattr(profile_objs[ep_id], "age", None),
                "risk_bucket": int(risk_bucket) if risk_bucket is not None else None,
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"overall": {"n_episodes": 0}, "by_risk_bucket": {}, "per_episode": []}

    def _agg(group):
        return {
            "n_episodes": int(len(group)),
            "avg_total_return": float(group["total_return"].mean()),
            "avg_max_drawdown": float(group["max_drawdown"].mean()),
            "avg_sharpe": float(group["sharpe"].mean()),
            "avg_cvar5": float(group["cvar5"].mean()),
        }

    overall = _agg(df)
    by_bucket: dict[str, Any] = {}
    if "risk_bucket" in df.columns:
        for bucket, sub in df.dropna(subset=["risk_bucket"]).groupby("risk_bucket"):
            by_bucket[str(int(bucket))] = _agg(sub)
    return {
        "overall": overall,
        "by_risk_bucket": by_bucket,
        "per_episode": df.to_dict(orient="records"),
        "meta": {
            "seed": actual_seed,
            "device": str(device),
            "observation_scaler": scaler_name,
            "reward_scale_applied": 0,
            "split": "val",
            "policy": policy_name,
        },
    }


def main() -> None:
    args = parse_args()

    if args.require_gpu:
        _require_gpu()

    is_fast = args.fast_dev or EXPERIMENT_MODE in _FAST_MODE_NAMES
    fqe_steps = args.fqe_steps
    eval_interval = args.eval_interval
    if is_fast:
        fqe_steps = min(fqe_steps, 50_000)
        eval_interval = min(eval_interval, 10_000)
        print(
            f"[eval_policy] fast_dev 模式启用 (mode={EXPERIMENT_MODE}) -> "
            f"FQE 步数={fqe_steps}，评估间隔={eval_interval}"
        )

    behavior_meta_path = _behavior_meta_path(args.dataset, args.behavior_meta)
    behavior_profiles = _load_behavior_profiles(behavior_meta_path)
    random_baseline_supported = False
    random_baseline_reason = "behavior meta unavailable"
    if behavior_meta_path and os.path.exists(behavior_meta_path):
        with np.load(behavior_meta_path, allow_pickle=True) as meta_tmp:
            has_action_sources = "action_sources" in meta_tmp.files
            allowed_counts = meta_tmp.get("allowed_action_counts")
        if has_action_sources:
            print("[eval_policy] 检测到 action_sources，可按来源分组风险指标。")
        else:
            print("[eval_policy] 行为 meta 缺少 action_sources，跳过按来源分组。")
        if allowed_counts is None:
            random_baseline_reason = "allowed_action_counts missing"
            print("[eval_policy] 警告：行为 meta 缺少 allowed_action_counts，随机基线加权跳过。")
        else:
            counts_array = np.asarray(allowed_counts, dtype=np.int32)
            if counts_array.size == 0 or np.all(counts_array <= 0):
                random_baseline_reason = "allowed_action_counts empty_or_nonpositive"
                print("[eval_policy] 警告：allowed_action_counts 为空或非正数，随机基线退化，忽略加权。")
            else:
                random_baseline_supported = True
                random_baseline_reason = "ok"
    else:
        print("[eval_policy] 行为 meta 不存在，无法检查 action_sources 与 allowed_action_counts。")

    model_path = args.model.strip()
    if model_path:
        train_cfg = load_training_config(model_path)
        reward_scale = train_cfg.get("reward_scale", 1.0)
    else:
        train_cfg = {}
        reward_scale = 1.0

    print(f"[eval_policy] 训练配置中 reward_scale={reward_scale}")

    replay_buffer = load_replay_buffer(args.dataset, reward_scale=reward_scale)
    applied_scale = getattr(replay_buffer, "_reward_scale_applied", None)
    print(f"[eval_policy] 数据集 reward_scale_applied={applied_scale}")
    if replay_buffer.episodes:
        sample_rewards = np.asarray(replay_buffer.episodes[0].rewards, dtype=np.float64)
        if sample_rewards.size:
            print(
                "[eval_policy] 样本奖励统计",
                f"最小={sample_rewards.min():.6f}",
                f"最大={sample_rewards.max():.6f}",
                f"均值={sample_rewards.mean():.6f}",
            )
    if behavior_profiles is None:
        print("[eval_policy] 行为 meta 缺少 user_profiles，分桶风险会丢失。")

    episode_bucket_map = _build_episode_bucket_map(
        behavior_profiles,
        len(replay_buffer.episodes),
    )
    dataset_info = getattr(replay_buffer, "dataset_info", None)
    dataset_action_size = getattr(dataset_info, "action_size", None)
    policy = None
    if model_path:
        policy = load_trained_policy(
            model_path,
            replay_buffer,
            require_gpu=args.require_gpu,
        )

    train_dataset, val_dataset = prepare_fqe_datasets(
        replay_buffer,
        validation_ratio=args.validation_ratio,
    )

    fqe_metrics: Dict[str, Any] = {}
    fqe_model = None
    if policy is not None:
        print("--- 运行 Fitted Q Evaluation (FQE) ---")
        fqe_metrics, fqe_model = run_fqe(
            policy,
            train_dataset,
            val_dataset,
            n_steps=fqe_steps,
            eval_interval=eval_interval,
            log_dir=args.log_dir,
            require_gpu=args.require_gpu,
        )
    else:
        print("[eval_policy] 未提供模型，跳过 FQE 评估。")

    fqe_by_bucket = _compute_fqe_by_bucket(fqe_model, replay_buffer, episode_bucket_map)
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
        "model_path": args.model,
        "dataset_path": args.dataset,
        "fqe": fqe_metrics,
        "cpe": cpe_metrics,
    }
    random_baseline_supported = "random_baseline_ips" in cpe_metrics
    summary["cpe_flags"] = cpe_metrics.get("cpe_flags")
    summary["random_baseline"] = {"supported": random_baseline_supported}
    if random_baseline_supported:
        summary["random_baseline"]["metrics"] = {
            "ips": cpe_metrics.get("random_baseline_ips"),
            "snips": cpe_metrics.get("random_baseline_snips"),
        }
    else:
        summary["random_baseline"]["reason"] = random_baseline_reason
    if fqe_by_bucket:
        summary["fqe_by_risk_bucket"] = fqe_by_bucket

    print("--- 统计 episode 风险指标（max drawdown） ---")
    episode_df = _collect_episode_metrics(
        replay_buffer,
        episode_bucket_map,
        reward_scale_fallback=reward_scale,
    )
    risk_behavior = _build_risk_metrics(episode_df)
    summary["risk_metrics_behavior"] = risk_behavior
    summary.pop("risk_metrics", None)
    print(
        "[eval_policy] NOTE: risk_metrics removed; use risk_metrics_behavior for baseline "
        "and backtest_metrics_policy for trained policy."
    )

    if args.backtest and args.policy == "trained" and policy is not None:
        print("--- 环境回测 ---")
        bt = run_backtest(
            policy,
            behavior_meta_path,
            args.dataset,
            os.path.basename(args.model) or "policy",
            action_size=dataset_action_size,
            max_episodes=args.backtest_episodes,
            behavior_profiles=behavior_profiles,
            seed=args.backtest_seed,
        )
        if bt:
            summary["backtest_metrics_policy"] = {
                k: v for k, v in bt.items() if k != "per_episode"
            }
            per_ep = bt.get("per_episode")
            backtest_csv = None
            if per_ep:
                df = pd.DataFrame(per_ep)
                model_dir = os.path.dirname(args.model) or "."
                os.makedirs(model_dir, exist_ok=True)
                out_csv = os.path.join(
                    model_dir,
                    f"{os.path.basename(args.model)}.backtest_val.csv",
                )
                df.to_csv(out_csv, index=False)
                backtest_csv = out_csv
                print(f"[eval_policy] 回测逐集 CSV 已写入 {out_csv}")
            if backtest_csv:
                summary["backtest_metrics_policy"]["csv_path"] = backtest_csv
        else:
            summary["backtest_metrics_policy"] = {}
        summary["backtest_metrics"] = summary["backtest_metrics_policy"]

    print("\n=== 评估结果 ===")
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(summary_text)
    if args.summary_output:
        out_path = args.summary_output
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write(summary_text)
        print(f"[eval_policy] 评估摘要已写入 {out_path}")


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
