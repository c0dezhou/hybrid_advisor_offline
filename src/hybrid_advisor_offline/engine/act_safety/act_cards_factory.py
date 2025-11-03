from typing import List

from .act_card_def import ActCard

def build_card_factory():
    """
    生成一个覆盖各种可能性的、更细粒度的行动“网格”，用于训练需要探索更多可能性的rl模型
    """
    cards: List[ActCard] = []
    aid = 0
    for eq in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for bd in [0.2, 0.3, 0.4, 0.5]:
            cs = round(1 - eq - bd, 2)
            if cs < 0:
                continue
            risk = 0 if eq <= 0.3 else (1 if eq <= 0.6 else 2)
            disclosure_key = {
                0: "CONSERVATIVE_STABILITY",
                1: "MODERATE_BALANCED",
                2: "AGGRESSIVE_GROWTH",
            }[risk]
            desc = f"目标权重≈ 股票{int(eq*100)}%/债{int(bd*100)}%/现金{int(cs*100)}%"
            cards.append(ActCard(
                act_id=aid,
                card_id=f"GRID_{int(eq*100)}_{int(bd*100)}_{int(cs*100)}",
                disclosure_key=disclosure_key,
                risk_level=risk,
                target_alloc=[eq, bd, cs],
                description=desc,
            ))
            aid += 1
    return cards
