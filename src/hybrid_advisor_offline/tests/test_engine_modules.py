import numpy as np
import pytest

from hybrid_advisor_offline.engine.act_safety.act_cards_factory import build_card_factory
from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import ALL_CARDS
from hybrid_advisor_offline.engine.policy import explain, policy_based_rule
from hybrid_advisor_offline.engine.personal import personal_prior
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
        reward_architect.compute_reward(0.1, 0.0, profile)


def test_build_explain_pack_text():
    class DummyCard:
        act_id = 7
        card_id = "AGG_GROWTH"
        disclosure_key = "AGGRESSIVE_GROWTH"

    pack = explain.build_explain_pack(DummyCard(), user_risk_bucket=2, model_version="demo")
    print(pack["customer_friendly_text"])
    print(pack["audit_text"])
    assert "进取型" in pack["customer_friendly_text"]


def _make_user(**overrides):
    defaults = dict(
        age=40,
        job="management",
        marital="married",
        education="tertiary",
        default="no",
        balance=5000,
        housing="no",
        loan="no",
    )
    defaults.update(overrides)
    return reward_architect.UserProfile(**defaults)


def test_compute_risk_aversion_rules():
    baseline = _make_user()
    assert reward_architect.compute_risk_aversion(baseline) == pytest.approx(1.0)

    aggressive = _make_user(age=25, balance=20_000)
    assert reward_architect.compute_risk_aversion(aggressive) == pytest.approx(0.72, rel=1e-3)

    conservative = _make_user(age=65, balance=200, housing="yes", loan="yes")
    # 1.3 (age) * 1.1 (balance) * 1.1 (housing) * 1.1 (loan) = 1.7323 -> clip 到 1.5
    assert reward_architect.compute_risk_aversion(conservative) == pytest.approx(1.5)


def test_reward_toggle(monkeypatch):
    profile = _make_user(age=65, balance=200, housing="yes", loan="yes")
    monkeypatch.setattr(reward_architect, "USE_PERSONAL_RISK_IN_REWARD", 1)
    reward_personal = reward_architect.compute_reward(
        0.01,
        0.05,
        profile,
        accept_prob=0.0,
    )

    monkeypatch.setattr(reward_architect, "USE_PERSONAL_RISK_IN_REWARD", 0)
    reward_uniform = reward_architect.compute_reward(
        0.01,
        0.05,
        profile,
        accept_prob=0.0,
    )

    assert reward_personal != reward_uniform


def test_drawdown_personalization(monkeypatch):
    risk_averse = _make_user(age=65, balance=200, housing="yes", loan="yes")
    risk_tolerant = _make_user(age=28, balance=20_000, housing="no", loan="no")
    monkeypatch.setattr(reward_architect, "USE_PERSONAL_RISK_IN_REWARD", 1)

    conservative_reward = reward_architect.compute_reward(
        market_return=0.0,
        drawdown=0.2,
        user_profile=risk_averse,
        accept_prob=0.0,
    )
    aggressive_reward = reward_architect.compute_reward(
        market_return=0.0,
        drawdown=0.2,
        user_profile=risk_tolerant,
        accept_prob=0.0,
    )
    assert conservative_reward < aggressive_reward


def test_drawdown_uniform_mode(monkeypatch):
    user_a = _make_user(age=65, balance=200, housing="yes", loan="yes")
    user_b = _make_user(age=28, balance=20_000, housing="no", loan="no")
    monkeypatch.setattr(reward_architect, "USE_PERSONAL_RISK_IN_REWARD", 0)
    monkeypatch.setattr(reward_architect, "UNIFORM_RA_FACTOR", 1.2)

    reward_a = reward_architect.compute_reward(
        market_return=0.0,
        drawdown=0.2,
        user_profile=user_a,
        accept_prob=0.0,
    )
    reward_b = reward_architect.compute_reward(
        market_return=0.0,
        drawdown=0.2,
        user_profile=user_b,
        accept_prob=0.0,
    )
    assert reward_a == pytest.approx(reward_b)


def test_personal_prior_switch(monkeypatch):
    cards = list(ALL_CARDS)
    allowed = [card.act_id for card in cards]

    monkeypatch.setattr(personal_prior, "USE_PERSONAL_PRIOR", False)
    zeros = personal_prior.build_personal_prior(
        allowed,
        prefs={"risk_hint": "aggressive"},
        risk_bucket=2,
    )
    assert set(zeros.keys()) == set(allowed)
    assert all(val == pytest.approx(0.0) for val in zeros.values())

    monkeypatch.setattr(personal_prior, "USE_PERSONAL_PRIOR", True)
    bumps = personal_prior.build_personal_prior(
        allowed,
        prefs={"risk_hint": "conservative", "horizon_years": 2},
        risk_bucket=2,
    )
    assert len(bumps) == len(allowed)
    assert max(bumps.values()) <= personal_prior.PRIOR_CAP + 1e-6
    assert min(bumps.values()) >= -personal_prior.PRIOR_CAP - 1e-6

    # 保守偏好应当更青睐低风险卡片
    low_risk = min(cards, key=lambda c: c.risk_level).act_id
    high_risk = max(cards, key=lambda c: c.risk_level).act_id
    assert bumps[low_risk] > bumps[high_risk]
