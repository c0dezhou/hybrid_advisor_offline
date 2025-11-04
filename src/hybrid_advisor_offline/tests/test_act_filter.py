import numpy as np
from hybrid_advisor_offline.engine.act_safety.act_filter import allowed_cards_for_user, apply_action

def test_act_filter():
    print("--- 测试动作过滤器 ---")
    
    # 保守型
    risk_bucket_0 = 0
    cards_0 = allowed_cards_for_user(risk_bucket_0)
    print(f"\n保守型用户的允许卡片 (风险等级 {risk_bucket_0}):")
    for card in cards_0:
        print(f"  - {card.card_id} (风险等级: {card.risk_level})")
    
    # 稳健型
    risk_bucket_1 = 1
    cards_1 = allowed_cards_for_user(risk_bucket_1)
    print(f"\n稳健型用户的允许卡片 (风险等级 {risk_bucket_1}):")
    for card in cards_1:
        print(f"  - {card.card_id} (风险等级: {card.risk_level})")

    # 进取型
    risk_bucket_2 = 2
    cards_2 = allowed_cards_for_user(risk_bucket_2)
    print(f"\n进取型用户的允许卡片 (风险等级 {risk_bucket_2}):")
    for card in cards_2:
        print(f"  - {card.card_id} (风险等级: {card.risk_level})")

    # 测试应用一个动作
    initial_allocation = np.array([1.0, 0.0, 0.0])
    selected_card = cards_1[0] # 选择一张稳健型卡片
    new_allocation = apply_action(initial_allocation, selected_card)
    print(f"\n将卡片 '{selected_card.card_id}' 应用于配置 {initial_allocation}...")
    print(f"  新的目标配置: {new_allocation}")

