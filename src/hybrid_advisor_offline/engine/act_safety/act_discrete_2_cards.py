import os
import math
from dataclasses import dataclass, field
from typing import List

# CON: Conservative (保守型)
# MOD: Moderate (稳健型)
# AGG: Aggressive (进取型)
@dataclass(frozen=True) # 让实例不可变，保证合规
class ActCard:
    """
    将投资动作离散化，合规化
    每个动作卡片都是合规可审计的，agent只能丛总选择
    """
    act_id: int # 动作的唯一的整数ID，供RL算法内部使用
    card_id: str # 人类可读的唯一卡片ID，如 "CON_CASH_HEAVY"。
    risk_level: int  # 风险等级（可根据业务需要修改，先暂时这么设置）0: 保守型, 1: 稳健型, 2: 进取型
    target_alloc: List[float] = field(repr=False) # 目标资产配置 [股票, 债券, 现金]
    description: str # 面向客户或理财经理的合规描述

    def __post_init__(self):
        # 后置初始化检查：确保目标资产配置的总和约等于1.0
        if not math.isclose(sum(self.target_alloc), 1.0, abs_tol=1e-5):
            raise ValueError(f"{self.card_id}的资产配置总和必须约为1")

# _DEFAULT_CARDS 这个变量是一个列表 (List)，
# 并且这个列表里面的每一个元素都应该是 ActCard 类的实例
_DEFAULT_CARDS: List[ActCard] = [
    # --- 风险等级 0: 保守型 ---
    ActCard(
        act_id=0,
        card_id="CON_CASH_HEAVY",
        risk_level=0,
        target_alloc=[0.10, 0.20, 0.70],  # 目标配置: [股票, 债券, 现金]
        description="优先考虑保本金，在当前市场环境下持有较高比例现金。",
    ),
    ActCard(
        act_id=1,
        card_id="CON_BOND_TILT",
        risk_level=0,
        target_alloc=[0.20, 0.40, 0.40],
        description="在保本基础上，适度增持一定数量的债券以寻求稳定收益。",
    ),

    # --- 风险等级 1: 稳健型 ---
    ActCard(
        act_id=2,
        card_id="MOD_BALANCED",
        risk_level=1,
        target_alloc=[0.40, 0.40, 0.20],
        description="维持股债均衡配置，以平衡市场波动与长期增长机会。",
    ),
    ActCard(
        act_id=3,
        card_id="MOD_GROWTH_TILT",
        risk_level=1,
        target_alloc=[0.50, 0.40, 0.10],
        description="在均衡基础上，略微提高风险资产比重，以提高收益。",
    ),

    # --- 风险等级 2: 进取型 ---
    ActCard(
        act_id=4,
        card_id="AGG_EQUITY_FOCUS",
        risk_level=2,
        target_alloc=[0.70, 0.20, 0.10],
        description="侧重于风险机会，旨在最大化长期资本增值潜力。",
    ),
    ActCard(
        act_id=5,
        card_id="AGG_FULL_EQUITY",
        risk_level=2,
        target_alloc=[0.80, 0.10, 0.10],
        description="进一步提高风险配置，以充分参与市场长期增长趋势。",
    ),
]


# 快速查找辅助函数，O(1) 时间复杂度
ALL_CARDS = _DEFAULT_CARDS
# act_id当键，card本身当值
_CARD_BY_ID_MAP = {card.act_id: card for card in ALL_CARDS}
_ACT_SPACE_SIZE = len(ALL_CARDS)

def get_card_by_id(act_id: int):
    """
    通过act_id查找一个actcard实例
    """
    if act_id not in _CARD_BY_ID_MAP:
        raise KeyError(f"无效的卡片id{act_id}")
    return _CARD_BY_ID_MAP[act_id]

def get_act_space_size():
    """
    返回已定义的离散动作的总数。
    """
    return _ACT_SPACE_SIZE

# # 示例：如何访问和使用动作卡片
# if __name__ == '__main__':
#     print(f"定义的动作卡片总数: {get_act_space_size()}")
    
#     # 获取一个特定的卡片
#     card = get_card_by_id(2)
#     print("\n卡片示例:")
#     print(f"  ID: {card.card_id}")
#     print(f"  风险等级: {card.risk_level}")
#     print(f"  目标配置 (股票, 债券, 现金): {card.target_alloc}")
#     print(f"  描述: {card.description}")