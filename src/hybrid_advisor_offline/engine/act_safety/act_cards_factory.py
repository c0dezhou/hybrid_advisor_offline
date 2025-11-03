from typing import List

from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import ActCard

# 生成一个覆盖各种可能性的、更细粒度的行动“网格”，用于训练需要探索更多可能性的rl模型
def build_card_factory():
    cards: List[ActCard] = []
    aid = 0
    for eq in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for bd in [0.2, 0.3, 0.4, 0.5]:
            cs = round(1 - eq - bd, 2)
            if cs < 0:
                continue
            risk = 0 if eq <= 0.3 else (1 if eq <= 0.6 else 2)
            desc = f"目标权重≈ 股票{int(eq*100)}%/债{int(bd*100)}%/现金{int(cs*100)}%"
            cards.append(ActCard(
                action_id=aid,
                card_id=f"GRID_{int(eq*100)}_{int(bd*100)}_{int(cs*100)}",
                risk_level=risk,
                target_alloc=[eq, bd, cs],
                description=desc,
            ))
            aid += 1
    return cards
