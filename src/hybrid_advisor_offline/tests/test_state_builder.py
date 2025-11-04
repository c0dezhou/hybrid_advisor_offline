import numpy as np
import pytest
from hybrid_advisor_offline.engine.state.state_builder import (
    build_state_vec,
    get_state_dim,
    MarketSnapshot,
    UserProfile,
    _user_model_pipeline,
)

@pytest.mark.skipif(_user_model_pipeline is None, reason="User model not trained.")
def test_state_builder():
    print("--- 测试状态构建器 ---")
    state_dimension = get_state_dim()
    print(f"动态确定的状态维度: {state_dimension}")

    # 创建样本输入
    mkt = MarketSnapshot(
        rolling_30d_returen=np.array([0.05, 0.01, 0.001]),
        rolling_30d_vol=np.array([0.15, 0.05, 0.02]),
        vix=0.18
    )
    user = UserProfile(age=30, job='technician', marital='single', education='secondary', default='no', balance=2000, housing='yes', loan='no')
    alloc = np.array([0.7, 0.2, 0.1])

    # 构建状态向量
    state_vec = build_state_vec(mkt, user, alloc)
    
    print(f"\n为风险等级为 {user.risk_bucket} (进取型) 的用户生成的状态向量:")
    print(f"  形状: {state_vec.shape}")
    print(f"  数据类型: {state_vec.dtype}")
    print(f"  前10个样本值: {state_vec[:10]}")
