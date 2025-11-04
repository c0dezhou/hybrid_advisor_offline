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

def compute_reward(
    market_return: float,
    user_profile: UserProfile,
    drawdown: float = 0.0
) -> float:
    """
    计算reward。
    """
    if _user_model_pipeline is None:
        raise RuntimeError(
            f"用户模型未找到 {USER_MODEL_PATH}。 "
            "请先train_user_model。"
        )

    # 1. 获取用户接受度概率
    user_df = user_profile.to_df()
    # predict_proba 返回 [class_0, class_1] 的概率，我们取 class_1 (接受) 的概率
    accept_prob = _user_model_pipeline.predict_proba(user_df)[0, 1]

    # 2. 计算复合奖励
    total_reward = (
        W_MKT_RETURN * market_return +
        W_USER_ACCEPT * accept_prob -
        W_DRAWDOWN_PENALTY * max(drawdown, 0)  # 仅在回撤为正时应用惩罚
    )
    
    return total_reward

