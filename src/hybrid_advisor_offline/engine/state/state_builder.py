# agent在做决策前，需要观察当前的环境状态。
# 所以我们需要把各种复杂的信息转换成一个纯数字的列表（向量）。
# 当前文件文件就是专门负责做这个转换工作的
import os
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass, fields, MISSING
from typing import Union, Optional

from hybrid_advisor_offline.offline.cql.train_usr_model import get_model_features

USER_MODEL_PATH = "./models/user_model.pkl"
INCLUDE_USER_IN_STATE = int(os.environ.get("INCLUDE_USER_IN_STATE", "1")) == 1

# 在模块加载时一次性加载用户模型pipeline和预处理器，以提高效率
try:
    _user_model_pipeline = joblib.load(USER_MODEL_PATH)
    _user_model_preprocessor = _user_model_pipeline.named_steps['preprocessor']
except FileNotFoundError:
    _user_model_pipeline = None
    _user_model_preprocessor = None

@dataclass # dataclass可以直接__eq__
class MarketSnapshot:
    """单个时间点上的市场特征"""
    rolling_30d_returen: np.ndarray # shape: (3,)，代表SPY, AGG, SHY的30天滚动回报
    rolling_30d_vol: np.ndarray # shape: (3,)，年化波动率
    vix: float # VIX恐慌指数，用来代替市场波动性

@dataclass
class UserProfile:
    """
    代表一个用户的画像，其字段与UCI银行营销数据集的列相对应。
    这里是实现个性化推荐的核心。
    """
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    # 用于预测的字段，在简单演示中可以有默认值
    contact: str = 'cellular' # 电话联系
    day_of_week: str = 'mon'
    month: str = 'may'
    duration: int = 150 # 沟通时长（s）
    campaign: int = 1 # 该客户在当前营销活动中的接触次数
    pdays: int = -1 # 距离上次接触该客户的天数（默认 - 1 = 无历史接触记录，行业通用标记）
    previous: int = 0 # 该客户在当前活动前的历史接触次数（默认 0 = 无接触历史
    poutcome: str = 'unknown' # 历史接触的结果（默认unknown= 未知，其他：success= 成功、failure= 失败）

    @property # 使该方法只读，像访问变量一样直接调用
    def risk_bucket(self) -> int:
        risk =0 if self.age >55 else (1 if self.age >35 else 2)
        return risk
    
    def to_df(self):
        """
        将用户画像转换为单行df，供给scikit-learn使用预处理pipeline
        """
        num_features, categori_features = get_model_features()
        all_features = num_features + categori_features
        
        # 对每条数据只保留需要的features，单元素转换为单列表
        profile_dict = {k: [v] for k,v in self.__dict__.items() if k in all_features}
        return pd.DataFrame(profile_dict)

# 原始输入：用户数据（user_row，dict/Series，可能有格式问题）；
# 数据清洗：取值→去空格 / 引号→类型转换（str→int/float）；
# 结构化：转为 UserProfile 实例（类型安全、字段完整）；
# 格式适配：UserProfile 实例 → DataFrame（适配预处理器输入）；
# 特征工程：预处理器 transform → 编码 / 标准化后的稀疏矩阵；
# 最终格式：稀疏矩阵 → 二维稠密数组 → 一维数组 → float32 类型；
# 输出：模型可直接接收的输入数组。
def user_row_to_profile(user_row: Union[pd.Series, dict]) -> UserProfile:
    """
    数据清洗和转换管道
    将bm_full.csv的数据的一行转换为以一个UP对象，填充默认值
    """
    if isinstance(user_row, pd.Series): # 将user row转换为dict
        row_data = user_row.to_dict()
    else:
        row_data = dict(user_row)

    normalized = {}
    for field_info in fields(UserProfile):#遍历UP中的所有字段，会筛选掉没有被定义的多余的列
        if not field_info.init:
            continue

        name = field_info.name
        if name in row_data:
            value = row_data[name]

            if pd.isna(value):# 值缺失，用默认数据填补
                if field_info.default is not MISSING:
                    value = field_info.default
                elif getattr(field_info, "default_factory", MISSING) is not MISSING:
                    value = field_info.default_factory()
                else:
                    # fallback兜底值
                    fallback = {str: "unknown", int: 0, float: 0.0}.get(field_info.type)
                    if fallback is None:
                        raise ValueError(f"'{name}'缺少默认值，需手动输入")
                    value = fallback
            else:
                # 1. 字符串格式清洗：去除首尾空格 + 去掉所有双引号
                if isinstance(value, str):
                    value = value.strip().replace('"', '')
                # 2. 按字段声明类型强制转换
                if field_info.type is int:
                    value = int(float(value)) # 先转float再转int，兼容字符串格式的数字（如"123.0"）
                elif field_info.type is float:
                    value = float(value)

            normalized[name] = value
        elif field_info.default is not MISSING:
            normalized[name] = field_info.default
        elif getattr(field_info, "default_factory", MISSING) is not MISSING:
            normalized[name] = field_info.default_factory()
        else:
            raise ValueError(f"数据行中缺少必要的用户字段 '{name}'。")
    return UserProfile(**normalized)


def make_up_to_vec(up: UserProfile):
    """
    用户画像转换为特征向量，迭代过程中使用，作为模型的输入格式
    """
    user_df = up.to_df()
    # 要展平可能之后用于拼接
    return _user_model_preprocessor.transform(user_df).toarray().flatten().astype(np.float32)

def _maybe_mask_user_vec(user_vec: np.ndarray) -> np.ndarray:
    """
    当 INCLUDE_USER_IN_STATE=0 时，将用户特征置零但保留原始维度。
    """
    if INCLUDE_USER_IN_STATE:
        return user_vec
    return np.zeros_like(user_vec, dtype=np.float32)

def build_state_vec(
    mkt_features: MarketSnapshot,
    user_profile: UserProfile,
    curr_alloc: np.ndarray,
    user_vector: Optional[np.ndarray] = None
):
    """
    构建agent的state vec。

    状态向量是以下部分的拼接：
    1. 市场特征 (回报率, 波动率)
    2. 经过pipeline处理的用户特征 
    3. 当前的投资组合配置

    """
    # 拼接市场特征
    mkt_vec = np.concatenate([
        mkt_features.rolling_30d_returen,
        mkt_features.rolling_30d_vol,
        np.array([mkt_features.vix])
    ]).astype(np.float32, copy=False)
    # print(f"DEBUG: mkt_vec shape={mkt_vec.shape}")

    # 处理用户特征
    if user_vector is None:
        user_vec = make_up_to_vec(user_profile)
    else:
        user_vec = np.asarray(user_vector, dtype=np.float32)
    # print(f"DEBUG: user_vec shape={user_vec.shape}")
    user_vec = _maybe_mask_user_vec(user_vec)

    # 处理当前资产配置
    alloc_vec = np.asarray(curr_alloc, dtype=np.float32)
    # print(f"DEBUG: alloc_vec shape={alloc_vec.shape}")

    # 拼接前三项
    state_vector = np.concatenate([
        mkt_vec,
        user_vec,
        alloc_vec
    ])

    # print(f"DEBUG: state_vector shape={state_vector.shape}")
    return state_vector

_state_dim = None
def get_state_dim() -> int:
    """
    动态计算状态向量的维度。
    通过构建一个虚拟的状态向量并返回其长度来避免使用“魔数”。
    """
    global _state_dim
    if _state_dim is not None:
        return _state_dim

    if _user_model_preprocessor is None:
        raise RuntimeError(
            f"无法确定状态维度，因为用户模型未加载。请train user model "
        )

    # 创建虚拟对象以推断维度
    dummy_mkt = MarketSnapshot(
        rolling_30d_returen=np.zeros(3),
        rolling_30d_vol=np.zeros(3),
        vix=0.0
    )
    dummy_user = UserProfile(
        age=40, job='management', marital='married', education='tertiary',
        default='no', balance=5000, housing='yes', loan='no'
    )
    dummy_alloc = np.array([0.6, 0.3, 0.1])

    # 构建一个样本状态向量以获取其大小
    dummy_state = build_state_vec(dummy_mkt, dummy_user, dummy_alloc)
    _state_dim = len(dummy_state)
    
    return _state_dim
