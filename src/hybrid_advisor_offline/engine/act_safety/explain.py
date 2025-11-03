import datetime
from typing import Dict

from .act_card_def import ActCard

# --- 这里应该设置经合规部门批准的解释模板 ---
# 这是一个字典，存储了所有面向客户的、预先批准的解释文本。
# 键 (如 "CONSERVATIVE_STABILITY") 与 ActCard 中的 disclosure_key 对应。
# 模板中的 {risk_profile} 会被动态替换为用户的风险等级名称。
EXPLAIN_TEMPLATES: Dict[str, str] = {
    "CONSERVATIVE_STABILITY": "根据您的[{risk_profile}]风险偏好，我们建议您稳健投资，多配置一些现金和债券来降低风险，以不变应万变。",
    "CONSERVATIVE_INCOME": "根据您的[{risk_profile}]风险偏好，我们建议在保本的基础上，多买点优质债券，这样能定期产生一些比较稳的现金收入。",
    "MODERATE_BALANCED": "根据您的[{risk_profile}]风险偏好，我们建议股票和债券均衡配置。这样既能控制风险，又不会错过市场上涨时赚钱的机会。",
    "MODERATE_GROWTH": "根据您的[{risk_profile}]风险偏好，这个配置在均衡的基础上，稍微多配置了一些股票等增长型资产，目的是抓住更多的市场增长机会。",
    "AGGRESSIVE_GROWTH": "根据您的[{risk_profile}]风险偏好，这个建议主要投资股票市场，目标是让您的资产能长期、快速地增值。",
    "AGGRESSIVE_MAX_GROWTH": "根据您的[{risk_profile}]风险偏好，这个配置在股票市场的投资比例非常高。我们希望最大程度上抓住市场长期增长的机会，当然，风险和波动也会更大。",
}

# 风险等级ID到人类可读名称的映射
RISK_BUCKET_NAMES: Dict[int, str] = {
    0: "保守型",
    1: "稳健型",
    2: "进取型",
}

# model_version: 做出推荐的策略模型的版本号。
def build_explain_pack(card: ActCard, user_risk_bucket: int, model_version: str = "cql_v1.0"):
    """
    为给定的动作生成一个符合合规要求的“解释包”。
    """
    risk_profile_name = RISK_BUCKET_NAMES.get(user_risk_bucket, "未知类型")
    
    # 1. 从模板生成面向客户的友好文本
    template = EXPLAIN_TEMPLATES.get(card.disclosure_key, "配置调整建议，请咨询您的理财经理。")
    customer_friendly_text = template.format(risk_profile=risk_profile_name) 

    # 2. 结构化的摘要，用于内部审计和日志记录，包含关键的决策上下文。
    timestamp = datetime.datetime.now().isoformat()
    audit_summary = (
        f"Timestamp: {timestamp}Z | " # Z即0时区，在日志和数据交换中，所有时间都使用一个标准时区（通常是UTC）
        f"ModelVersion: {model_version} | "
        f"UserRiskBucket: {user_risk_bucket} ({risk_profile_name}) | "
        f"ChosenActionID: {card.act_id} | "
        f"ChosenCardID: {card.card_id} | "
        f"DisclosureKey: {card.disclosure_key}"
    )

    return {
        "customer_friendly_text": customer_friendly_text,
        "audit_text": audit_summary,
    }

if __name__ == '__main__':
    from .act_discrete_2_cards import get_card_by_id

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
