from typing import List
import numpy as np

from .act_card_def import ActCard
from .act_discrete_2_cards import ALL_CARDS

# 注意：user_risk_bucket代表用户的风险等级（用户属性），risk_level代表投资策略（Action Card）的风险等级。（投资策略属性）
def allowed_cards_for_user(user_risk_bucket: int):
    """
    根据用户的风险等级，筛选出该用户被允许执行的“动作卡片”列表。
    一个投资策略的风险等级，必须小于或等于用户自身的风险等级（可能经过动态调整后）
    这个函数是一个前置的合规性关卡，确保强化学习智能体只能从一个预先批准的、
    风险适当的动作集合中进行选择。
    """
    if user_risk_bucket not in [0, 1, 2]:
        # user_risk_bucket (int): 用户的风险承受能力等级 (0: 保守型, 1: 稳健型, 2: 进取型)。
        raise ValueError("用户风险等级(user_risk_bucket)必须是 0, 1, 或 2。")

    # 规则是：一个用户可以被推荐其风险等级或更低等级的卡片。
    allowed = [card for card in ALL_CARDS if card.risk_level <= user_risk_bucket]

    # 安全回退机制：如果由于某种原因（例如配置错误）没有可用的卡片，
    # 默认返回最保守的卡片，以确保绝对安全。
    if not allowed:
        allowed = [card for card in ALL_CARDS if card.risk_level == 0]
        
    return allowed

# current_alloc未来可能用于：
# 计算差异：diff = card.target_alloc - current_alloc，以确定需要买入和卖出哪些资产。
# 考虑交易成本：基于 diff 计算交易成本，并从最终的配置或现金中扣除。
# 模拟渐进式调整：可能不会一步到位，而是设定一个调整速度，例如 new_alloc = current_alloc + speed * diff。
# 设置阈值：如果 current_alloc 和 card.target_alloc 已经非常接近，则可能决定不执行任何操作以避免不必要的微小交易。
def apply_action(current_alloc: np.ndarray, card: ActCard):
    """
    将选定的动作卡片应用到当前的资产配置上。
    
    在这个简化的模型中，我们假设资产配置会瞬间调整到卡片的目标配置。
    可以不必关心交易执行的复杂细节（如交易成本、滑点、市场冲击等）
    """
    # 动作的执行就是采纳所选卡片的目标配置。
    return np.array(card.target_alloc, dtype=np.float32)
