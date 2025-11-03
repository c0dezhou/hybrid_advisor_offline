import math
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True) # 让实例不可变，保证合规
class ActCard:
    """
    将投资动作离散化，合规化
    每个动作卡片都是合规可审计的，agent只能丛总选择
    """
    act_id: int # 动作的唯一的整数ID，供RL算法内部使用
    card_id: str # 人类可读的唯一卡片ID，如 "CON_CASH_HEAVY"。
    disclosure_key: str # 合规披露的key
    risk_level: int  # 风险等级（可根据业务需要修改，先暂时这么设置）0: 保守型, 1: 稳健型, 2: 进取型
    target_alloc: List[float] = field(repr=False) # 目标资产配置 [股票, 债券, 现金]
    description: str # 面向客户或理财经理的合规描述

    def __post_init__(self):
        # 后置初始化检查：确保目标资产配置的总和约等于1.0
        if not math.isclose(sum(self.target_alloc), 1.0, abs_tol=1e-5):
            raise ValueError(f"{self.card_id}的资产配置总和必须约为1")
