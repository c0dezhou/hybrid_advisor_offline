# 使用 DiscreteFQE + InitialStateValueEstimationEvaluator / AverageValueEstimationEvaluator 
# 进行 FQE，并在结束后给出训练/验证指标

from typing import Dict, Optional

import numpy as np

from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.logging import FileAdapterFactory
from d3rlpy.metrics import (
    AverageValueEstimationEvaluator,
    InitialStateValueEstimationEvaluator,
)


def _mean_episode_length(dataset: Optional[MDPDataset]) -> Optional[float]:
    if dataset is None:
        return None
    lengths = [ep.transition_count for ep in dataset.episodes]
    if not lengths:
        return None
    return float(np.mean(lengths))


def run_fqe(
    policy,
    train_dataset: MDPDataset,
    val_dataset: Optional[MDPDataset] = None,
    *,
    n_steps: int = 100_000,
    eval_interval: int = 10_000,
    log_dir: str = "d3rlpy_logs/fqe",
    require_gpu: bool = False,
):
    """
    使用 d3rlpy 的 DiscreteFQE 对策略进行拟合 Q 评估。
    """
    config = FQEConfig()
    device = 0 if require_gpu else False
    fqe = DiscreteFQE(policy, config=config, device=device)

    scorers = {
        "initial_state_value": InitialStateValueEstimationEvaluator(),
        "average_value": AverageValueEstimationEvaluator(),
    }

    logger = FileAdapterFactory(root_dir=log_dir)
    fqe.fit(
        train_dataset,
        n_steps=n_steps,
        logger_adapter=logger,
        show_progress=True,
        n_steps_per_epoch=eval_interval,
    )

    init_eval: InitialStateValueEstimationEvaluator = scorers["initial_state_value"]  # type: ignore[assignment]
    avg_eval: AverageValueEstimationEvaluator = scorers["average_value"]  # type: ignore[assignment]

    metrics: Dict[str, float] = {
        "train_initial_state_value": init_eval(fqe, train_dataset),
        "train_average_value": avg_eval(fqe, train_dataset),
    }

    if val_dataset is not None: # 在验证集上计算
        metrics["val_initial_state_value"] = init_eval(fqe, val_dataset)
        metrics["val_average_value"] = avg_eval(fqe, val_dataset)

    discount = getattr(getattr(fqe, "impl", None), "gamma", 0.99)

    def _append_scaled_metrics(prefix: str, dataset: Optional[MDPDataset]) -> None:
        avg_value_key = f"{prefix}_average_value"
        if avg_value_key not in metrics:
            return
        avg_reward = (1.0 - discount) * metrics[avg_value_key]
        metrics[f"{prefix}_avg_reward_per_step"] = avg_reward
        mean_len = _mean_episode_length(dataset)
        if mean_len is not None:
            metrics[f"{prefix}_avg_episode_length"] = mean_len
            metrics[f"{prefix}_est_episode_return"] = avg_reward * mean_len

    _append_scaled_metrics("train", train_dataset)
    _append_scaled_metrics("val", val_dataset)

    return metrics, fqe
