import numpy as np
from typing import List, Tuple

from hybrid_advisor_offline.engine.act_safety.act_card_def import ActCard

# 定义了判断市场是否处于“高波动性”状态的阈值。
# 如果从状态向量中提取出的 VIX 水平（或其代理）超过这个值，就被认为是高波动性市场。
# 这是一个简化处理；真实系统可能会使用更动态的阈值。
HIGH_VOLATILITY_THRESHOLD = 0.20

# 规则1：如果市场波动率（VIX）非常高，就采取“保守”的动作（比如，把大部分资产换成现金）。
# 规则2：如果用户的风险偏好是“进取型”，并且近期市场回报率为正，就采取“进取”的动作（比如，增加股票配置）。
# 规则3：如果用户的风险偏好是“保守型”，则无论市场如何，都采取“保守”的动作。

# 该policy作用：
# 作为Baseline：为更复杂的强化学习（RL）智能体提供一个性能比较的基准。如果RL智能体的表现不如这个简单的规则策略，那就说明RL模型可能存在问题。
# 行为策略（Behavior Policy）：在离线强化学习中，它可以用来生成数据集。这些数据集包含了由这个规则策略在环境中采取行动所产生的状态、动作和奖励，然后可以用于训练一个更复杂的RL智能体。
# 快速验证：在环境开发早期，可以用它来快速测试环境的逻辑是否正确。

# state_vec (np.ndarray): 完整的状态向量。假设其布局为：
#                    [市场特征(7), 用户特征(N), 当前配置(3)]。
#                    7个市场特征是：
#                    - 30天滚动回报 (3)
#                    - 30天滚动波动率 (3)
#                    - VIX水平 (1)
def policy_based_rule(
    state_vec: np.ndarray,
    allowed_cards: List[ActCard], # 已根据用户风险等级过滤的 `ActCard` 列表
    user_risk_bucket: int
):
    """
    一个基于规则的策略，用于生成baseline行为并用于回测比较。
    这个策略也作为“行为策略”，用于生成离线数据集。
    """
    if not allowed_cards:
        raise ValueError("没有可用动作，允许的卡片列表(allowed_cards)不能为空。")

    # 从状态向量中提取 'vix'。它是第7个元素（索引为6）。
    # state_vec 布局: [回报(3), 波动率(3), vix(1), 用户向量(N), 配置(3)]
    vix = state_vec[6]

    if vix > HIGH_VOLATILITY_THRESHOLD:
        # 高波动性：采取防御姿态。选择风险等级最低的卡片。
        # 合规术语：“在高波动性环境下，我们通过选择最稳定的可用配置来优先保护资本。”
        chosen_card = min(allowed_cards, key=lambda card: card.risk_level)
    else:
        # 正常波动性：采取机会主义姿态。选择风险等级最高的卡片。
        # 合规术语：“在稳定的市场环境中，我们旨在通过选择与客户风险承受能力上限
        # 一致的配置来捕捉增长机会。”
        chosen_card = max(allowed_cards, key=lambda card: card.risk_level)
    
    return chosen_card.action_id, chosen_card