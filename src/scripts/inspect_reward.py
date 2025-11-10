from hybrid_advisor_offline.engine.envs.market_envs import MarketEnv
from hybrid_advisor_offline.engine.rewards.reward_architect import compute_reward, compute_risk_aversion
from hybrid_advisor_offline.engine.state.state_builder import user_row_to_profile
import pandas as pd

user_df = pd.read_csv("data/bm_full.csv", sep=";")
conservative = user_row_to_profile(user_df.iloc[0])
aggressive = user_row_to_profile(user_df.iloc[1])
env = MarketEnv(max_episode_steps=5)

for profile in (conservative, aggressive):
    info = env.reset()
    ra = compute_risk_aversion(profile)
    print(f"\n用户 {profile.age}/{profile.balance}, risk_aversion={ra:.2f}")
    for step in range(5):
        market_return, done, next_info = env.step(0)
        reward = compute_reward(market_return, next_info["drawdown"], profile, accept_prob=0.0)
        print(f"step={step:02d} ret={market_return:+.4f} drawdown={next_info['drawdown']:.4f} reward={reward:+.4f}")
        if done: break
        info = next_info
