from hybrid_advisor_offline.engine.policy.explain import build_explain_pack
from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_card_by_id

def test_explain_pack_builder():
    print("--- 测试解释包生成器 ---")

    # 示例 1: 一个稳健型用户得到了一个均衡型卡片
    moderate_user_bucket = 1
    balanced_card = get_card_by_id(2) # MOD_BALANCED
    
    pack1 = build_explain_pack(balanced_card, moderate_user_bucket)
    
    print(f"\n场景 1: 稳健型用户, 均衡型卡片 ({balanced_card.card_id})")
    print(f"  客户文本: {pack1['customer_friendly_text']}")
    print(f"  审计文本: {pack1['audit_text']}")

    # 示例 2: 一个进取型用户得到了一个增长型卡片
    aggressive_user_bucket = 2
    growth_card = get_card_by_id(4) # AGG_EQUITY_FOCUS

    pack2 = build_explain_pack(growth_card, aggressive_user_bucket, model_version="cql_v1.1-beta")

    print(f"\n场景 2: 进取型用户, 增长型卡片 ({growth_card.card_id})")
    print(f"  客户文本: {pack2['customer_friendly_text']}")
    print(f"  审计文本: {pack2['audit_text']}")

