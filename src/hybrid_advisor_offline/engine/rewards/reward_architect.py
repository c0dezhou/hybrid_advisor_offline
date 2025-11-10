import os
import joblib

from hybrid_advisor_offline.engine.state.state_builder import UserProfile

USER_MODEL_PATH = "./models/user_model.pkl"


def _get_weight(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        print(f"[reward_architect] 环境变量 {name}={val} 非法，回退到默认 {default}")
        return default


# reward 各维权重，可通过环境变量微调，方便实验切换
W_MKT_RETURN = _get_weight("W_MKT_RETURN", 1.0)
W_USER_ACCEPT = _get_weight("W_USER_ACCEPT", 0.0)
# V2.2.x: 调低回撤惩罚强度，避免 reward 全线变成巨量负值
W_DRAWDOWN_PENALTY = _get_weight("W_DRAWDOWN_PENALTY", 0.1)
MAX_DRAWDOWN_FOR_PENALTY = _get_weight("MAX_DRAWDOWN_FOR_PENALTY", 0.2)
USE_PERSONAL_RISK_IN_REWARD = int(os.getenv("USE_PERSONAL_RISK_IN_REWARD", "1"))
# 统一 reward 分支下的“固定”风险厌恶，用于快速降低或放大回撤惩罚
UNIFORM_RA_FACTOR = _get_weight("UNIFORM_RISK_AVERSION", 1.0)
# 简单的 debug 限流计数，用于打印 reward 诊断
_REWARD_DEBUG_LIMIT = int(os.getenv("REWARD_DEBUG", "0"))
_reward_debug_hits = 0

print(
    f"[reward_architect] 当前 reward 权重: "
    f"W_MKT_RETURN={W_MKT_RETURN}, "
    f"W_USER_ACCEPT={W_USER_ACCEPT}, "
    f"W_DRAWDOWN_PENALTY={W_DRAWDOWN_PENALTY} (cap={MAX_DRAWDOWN_FOR_PENALTY}), "
    f"USE_PERSONAL_RISK_IN_REWARD={USE_PERSONAL_RISK_IN_REWARD}, "
    f"UNIFORM_RISK_AVERSION={UNIFORM_RA_FACTOR}"
)

try:
    _user_model_pipeline = joblib.load(USER_MODEL_PATH)
except FileNotFoundError:
    _user_model_pipeline = None


def compute_risk_aversion(user_profile: UserProfile) -> float:
    """
    根据 UCI 用户画像计算风险厌恶系数，并限制在 [0.7, 1.5] 内。

    > 1.0 更保守，会加大回撤惩罚；
    < 1.0 更激进，会减弱回撤惩罚。
    """
    factor = 1.0

    age = getattr(user_profile, "age", None)
    balance = getattr(user_profile, "balance", None)
    housing = getattr(user_profile, "housing", None)
    loan = getattr(user_profile, "loan", None)

    if age is not None:
        if age >= 55:
            factor *= 1.3
        elif age <= 30:
            factor *= 0.8

    if balance is not None:
        if balance < 500:
            factor *= 1.1
        elif balance > 10_000:
            factor *= 0.9

    if isinstance(housing, str) and housing.lower() == "yes":
        factor *= 1.1
    if isinstance(loan, str) and loan.lower() == "yes":
        factor *= 1.1

    return float(max(0.7, min(factor, 1.5)))

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
    drawdown: float,
    user_profile: UserProfile,
    accept_prob: float | None = None,
):
    """
    计算 reward 的核心入口。
    - market_return：市场收益，越大越好；
    - drawdown：最大回撤，只罚正数；
    - user_profile：用于读取 risk aversion（只有个性化 reward 打开时才使用）；
    - accept_prob：预先算好的“客户接受概率”，可重复利用，避免每步跑模型。
    """
    if accept_prob is None:
        accept_prob = get_accept_prob(user_profile)

    eff_drawdown = max(drawdown, 0.0)
    if MAX_DRAWDOWN_FOR_PENALTY > 0.0:
        # V2.2.x: 只按限定最大回撤惩罚，超出部分不再继续线性放大
        eff_drawdown = min(eff_drawdown, MAX_DRAWDOWN_FOR_PENALTY)

    penalty = W_DRAWDOWN_PENALTY * eff_drawdown
    active_mode = "personal" if USE_PERSONAL_RISK_IN_REWARD else "uniform"
    if USE_PERSONAL_RISK_IN_REWARD:
        risk_aversion = compute_risk_aversion(user_profile)
        penalty *= risk_aversion
        applied_ra = risk_aversion
    else:
        # 统一 reward 模式下使用固定系数，避免每次都去加载用户画像
        penalty *= UNIFORM_RA_FACTOR
        applied_ra = UNIFORM_RA_FACTOR

    total_reward = (
        W_MKT_RETURN * market_return +
        W_USER_ACCEPT * accept_prob -
        penalty  # drawdown>0 时生效
    )
    _maybe_log_reward(market_return, eff_drawdown, applied_ra, total_reward, active_mode)
    return total_reward


def _maybe_log_reward(
    market_return: float,
    eff_drawdown: float,
    risk_aversion: float,
    total_reward: float,
    mode: str,
) -> None:
    global _reward_debug_hits
    if _REWARD_DEBUG_LIMIT <= 0 or _reward_debug_hits >= _REWARD_DEBUG_LIMIT:
        return
    print(
        f"[reward_debug] mode={mode} "
        f"market_return={market_return:.6f} "
        f"eff_drawdown={eff_drawdown:.6f} "
        f"risk_aversion={risk_aversion:.3f} "
        f"reward={total_reward:.6f}"
    )
    _reward_debug_hits += 1
