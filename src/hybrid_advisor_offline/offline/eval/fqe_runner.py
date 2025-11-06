# 使用 DiscreteFQE + InitialStateValueEstimationEvaluator / AverageValueEstimationEvaluator 
# 进行 FQE，并在结束后给出训练/验证指标

from typing import Dict, Optional

from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.logging import FileAdapterFactory
from d3rlpy.metrics import (
    AverageValueEstimationEvaluator,
    InitialStateValueEstimationEvaluator,
)


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

    return metrics
