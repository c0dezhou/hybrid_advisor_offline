import os
import joblib
import numpy as np

from hybrid_advisor_offline.engine.state.state_builder import UserProfile

USER_MODEL_PATH = "./models/user_model.pkl"

W_MKT_RETURN = 1.0  # 市场回报权重
W_USER_ACCEPT = 0.5    # 用户接受度权重
W_DRAWDOWN_PENALTY = 0.2 # 最大回撤惩罚权重

try:
    _user_model_pipeline = joblib.load(USER_MODEL_PATH)
except FileNotFoundError:
    _user_model_pipeline = None

def _ensure_model_loaded():
    if _user_model_pipeline is None:
        raise RuntimeError(
            f"用户模型未找到 {USER_MODEL_PATH}。 "
            "请先train_user_model。"
        )


# 在 reward_architect.py 新增 get_accept_prob，
# 专门负责把用户画像转换成接受概率；compute_reward 增加 accept_prob 参数，
# 允许外部把已经算好的概率传回来，这样不必在循环里重复跑模型
def get_accept_prob(user_profile: UserProfile):
    """
    计算给定用户画像的接受概率，供奖励函数或其他模块复用。
    """
    _ensure_model_loaded()
    user_df = user_profile.to_df()
    return float(_user_model_pipeline.predict_proba(user_df)[0, 1])


def compute_reward(
    market_return: float,
    user_profile: UserProfile,
    drawdown: float = 0.0,
    accept_prob: float | None = None,
):
    """
    计算复合奖励。支持传入预先计算的 `accept_prob` 以避免重复模型推理。
    """
    if accept_prob is None:
        accept_prob = get_accept_prob(user_profile)

    # 1. 获取用户接受度概率
    # predict_proba 返回 [class_0, class_1] 的概率，我们取 class_1 (接受) 的概率
    total_reward = (
        W_MKT_RETURN * market_return +
        W_USER_ACCEPT * accept_prob -
        W_DRAWDOWN_PENALTY * max(drawdown, 0)  # 仅在回撤为正时应用惩罚
    )
    
    return total_reward
