import pytest
from hybrid_advisor_offline.engine.state.state_builder import UserProfile
from hybrid_advisor_offline.engine.rewards.reward_architect import compute_reward, _user_model_pipeline

def test_compute_reward():
    if _user_model_pipeline:
        print("------测试奖励计算器------")

        # 创建一个示例用户
        user = UserProfile(
            age=45, job='management', marital='married', education='tertiary',
            default='no', balance=10000, housing='yes', loan='no'
        )
        
        # 场景 1: 市场正回报，无回撤
        market_ret_1 = 0.005 # +0.5%
        reward_1 = compute_reward(market_ret_1, 0.0, user)
        print(f"\n场景 1: 市场正回报 ({market_ret_1*100:.2f}%)")
        print(f"  计算出的奖励: {reward_1:.4f}")

        # 场景 2: 市场负回报，有回撤
        market_ret_2 = -0.01 # -1.0%
        drawdown_2 = 0.05   # 5% 回撤
        reward_2 = compute_reward(market_ret_2, drawdown_2, user)
        print(f"\n场景 2: 市场负回报 ({market_ret_2*100:.2f}%) 伴随回撤 ({drawdown_2*100:.1f}%)")
        print(f"  计算出的奖励: {reward_2:.4f}")
        
        # 获取用户接受度概率作为上下文
        prob = _user_model_pipeline.predict_proba(user.to_df())[0, 1]
        print(f"\n(该用户画像的接受度概率: {prob:.4f})")
    else:
        pytest.skip("无法运行测试，因为用户模型尚未训练。")
