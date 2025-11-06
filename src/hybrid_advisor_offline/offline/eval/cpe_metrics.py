
# 行为策略离线评估（CPE）指标。

"""
IPS / SNIPS：两种常用的「离线策略评估方法」（用来在离线数据集上估算一个新策略的收益，不用实际部署）。核心是通过「倾向得分」修正数据偏差，公式简化理解：
IPS（逆倾向得分加权）：收益估算 = Σ(实际收益 / 倾向得分) / 样本数
SNIPS（标准化 IPS）：在 IPS 基础上增加了 “新策略选择该动作的概率” 权重，修正更精准
倾向得分（propensity）：离线数据集中，「原始策略（比如之前的旧策略）选择某个动作的概率」（比如用户点击 A 按钮的概率是 0.3，这个 0.3 就是倾向得分）。
on-policy 平均：直接对数据集中的收益取平均值（等价于假设 “倾向得分 = 1.0”），只有当数据集是「当前策略自己产生的」（on-policy）时，这样估算才无偏差；如果是用其他策略的数据集评估新策略（off-policy），直接平均会有偏差。
"""

# 我们现在要统计 policy_based_rule 的平均收益，用 IPS/SNIPS 方法来算，
# 但有个前提 —— 离线数据集里没明确存 “倾向得分”（比如没记录原始规则策略选每个动作的概率）。
# 这种情况下，允许「默认把倾向得分设为 1.0」，此时 IPS/SNIPS 就退化成了 “直接算数据里的收益平均值”（on-policy 平均）。
# 等以后数据集完善了，只要里面有 “倾向得分”（可以存在transition.info["propensity"]字段，或其他存储概率的地方）
# 系统就会自动用这个真实的倾向得分来计算，不用手动改逻辑。


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from d3rlpy.dataset import ReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class CPESample:
    reward: float
    propensity: float


def collect_cpe_samples(buffer: ReplayBuffer):
    """从 ReplayBuffer 中提取奖励与 propensity。"""
    samples: List[CPESample] = []
    for episode in buffer.episodes:
        rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
        for reward in rewards:
            samples.append(CPESample(reward=float(reward), propensity=1.0))
    return samples


def ips(samples: Iterable[CPESample]):
    """重要性采样 (Importance Sampling) 评估器"""
    num = 0.0
    den = 0.0
    for s in samples:
        w = 1.0 / s.propensity
        num += w * s.reward
        den += w
    return num / max(den, 1e-9)


def snips(samples: Iterable[CPESample]):
    """自归一化重要性采样 (Self-Normalized Importance Sampling)"""
    sample_list = list(samples)
    weights = [1.0 / s.propensity for s in sample_list]
    total_w = sum(weights) or 1.0
    return sum(w * s.reward for w, s in zip(weights, sample_list)) / total_w


def mean_episode_return(buffer: ReplayBuffer):
    """计算数据集中所有轨迹的平均总回报"""
    returns = []
    for episode in buffer.episodes:
        rewards = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
        returns.append(float(np.sum(rewards)))
    if not returns:
        return 0.0
    return float(np.mean(returns))


def compute_cpe_report(buffer: ReplayBuffer):
    """
    计算离线日志上的 IPS / SNIPS / 平均回报等指标。
    """
    samples = collect_cpe_samples(buffer)
    if not samples:
        logger.warning("CPE 样本为空，返回 0 指标。")
        return {"episode_return_mean": 0.0, "ips": 0.0, "snips": 0.0}

    if any(abs(s.propensity - 1.0) > 1e-6 for s in samples):
        logger.info("检测到非 1.0 的 propensity，使用提供的行为概率。")
    else:
        logger.info(
            "propensity 信息缺失，默认为 1.0 —— 结果等价于 on-policy 平均。"
        )

    return {
        "episode_return_mean": mean_episode_return(buffer),
        "ips": ips(samples),
        "snips": snips(samples),
    }
