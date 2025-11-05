import numpy as np
import pytest

from hybrid_advisor_offline.engine.act_safety.act_cards_factory import build_card_factory
from hybrid_advisor_offline.engine.policy import explain, policy_based_rule
from hybrid_advisor_offline.engine.rewards import reward_architect


def test_build_card_factory_basic():
    cards = build_card_factory()
    print(f"factory 卡片数量: {len(cards)}")
    assert len(cards) > 0


def test_policy_based_rule_paths():
    state_vec = np.zeros(10, dtype=np.float32)
    low_risk = type("Card", (), {"risk_level": 0, "act_id": 1, "action_id": 1, "card_id": "LOW"})()
    high_risk = type("Card", (), {"risk_level": 2, "act_id": 2, "action_id": 2, "card_id": "HIGH"})()

    state_vec[6] = 0.5  # 高波动
    _, chosen_high_vol = policy_based_rule.policy_based_rule(state_vec, [low_risk, high_risk], 1)
    assert chosen_high_vol.card_id == "LOW"

    state_vec[6] = 0.1  # 正常波动
    _, chosen_normal = policy_based_rule.policy_based_rule(state_vec, [low_risk, high_risk], 2)
    assert chosen_normal.card_id == "HIGH"


def test_policy_based_rule_no_cards():
    state_vec = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError):
        policy_based_rule.policy_based_rule(state_vec, [], 0)


def test_compute_reward_without_model(monkeypatch):
    monkeypatch.setattr(reward_architect, "_user_model_pipeline", None)
    profile = reward_architect.UserProfile(
        age=40,
        job="management",
        marital="married",
        education="tertiary",
        default="no",
        balance=5000,
        housing="yes",
        loan="no",
    )
    with pytest.raises(RuntimeError):
        reward_architect.compute_reward(0.1, profile, drawdown=0.0)


def test_build_explain_pack_text():
    class DummyCard:
        act_id = 7
        card_id = "AGG_GROWTH"
        disclosure_key = "AGGRESSIVE_GROWTH"

    pack = explain.build_explain_pack(DummyCard(), user_risk_bucket=2, model_version="demo")
    print(pack["customer_friendly_text"])
    print(pack["audit_text"])
    assert "进取型" in pack["customer_friendly_text"]
