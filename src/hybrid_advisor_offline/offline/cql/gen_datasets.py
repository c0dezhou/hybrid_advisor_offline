# fetch data
import io
import os
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ucimlrepo import fetch_ucirepo

import sys
from tqdm import tqdm
from d3rlpy.dataset import MDPDataset
from d3rlpy.constants import ActionSpace

from hybrid_advisor_offline.engine.envs.market_envs import MarketEnv
from hybrid_advisor_offline.engine.state.state_builder import (
    UserProfile,
    build_state_vec,
    get_state_dim,
    user_row_to_profile,
    make_up_to_vec,
)
from hybrid_advisor_offline.engine.rewards.reward_architect import (
    compute_reward,
    get_accept_prob,
)
from hybrid_advisor_offline.engine.act_safety.act_filter import allowed_cards_for_user
from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_act_space_size
from hybrid_advisor_offline.engine.policy.policy_based_rule import policy_based_rule


DATA_DIR = "./data"
# SPY：标普 500 指数 ETF
# AGG：美国综合债券 ETF
# SHY：短期国债 ETF 使用SHY作为现金的模拟。原因：1.模拟无风险利率，2.可交易性 3.真实市场数据驱动
_STOOQ_MAP = {"SPY": "spy.us", "AGG": "agg.us", "SHY": "shy.us"}

USER_DATA_FILE = "./data/bm_full.csv"  # 输入的用户数据文件
OUTPUT_DATASET_PATH = "./data/offline_dataset.h5"  # 输出的数据集文件
N_USERS_TO_SIMULATE = 500  # 用于模拟的用户数量,生成500条轨迹

def _fetch_data_from_yf(ticker, maxretry, base_sleep_second):
    for attempt in range(1, maxretry + 1):
        try:
            data = yf.download(
                ticker,
                period = "10y",
                interval = "1d",
                progress=True, # 显示下载进度条看看
                auto_adjust=False, # 不调整
                threads=True, # 启用多线程
            )
            if data.empty or "Adj Close" not in data.columns:
                raise ValueError(f"{ticker} 的数据是空的 or 缺少 'Adj Close' 列。")
            return data["Adj Close"].rename(ticker)
        except Exception as exc:
            if attempt == maxretry:
                raise RuntimeError(f"获取{ticker}失败") from exc
            sleep_sec = base_sleep_second*attempt
            print(f"yf err: {exc}, 尝试{attempt}/{maxretry}, {sleep_sec}s后重试....")
            time.sleep(sleep_sec)
    raise RuntimeError(f"yf 尝试太多次{ticker}")

# 如果yf get失败，从stooq get数据，再失败就创建合成数据
def _fetch_data_from_stooq(ticker):
    symbol = _STOOQ_MAP.get(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout = 35, headers={"User-Agent": "Mozilla/5.0"})# 绕过反爬
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text)) #io.StringIO创建虚拟文本流，从虚拟文件读数据
    if df.empty or "Close" not in df.columns:
        raise RuntimeError(f"stooq返回{ticker}空")
    # rename(ticker): 将这一列数据命名为对应的股票代码（如 "SPY"）
    s = df.set_index("Date")["Close"].rename(ticker).sort_index()
    s.index = pd.to_datetime(s.index)
    return s

# 模拟生成一支股票在指定工作日内的价格走势，基于正态分布的每日收益率和复利计算，最终返回一个带有日期索引的价格序列。
def _gen_synthetic_series(ticker):
    base_price = {"SPY": 100.0, "AGG": 105.0, "SHY": 99.0}.get(ticker, 100.0)
    days = 2520 # 10年的工作日
    date_idx = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    # X % N 的结果永远在 0 到 N-1 之间。
    # % (2 ** 32) 能确保最终得到的数字一定在 0 到 2**32 - 1 这个区间内
    rng = np.random.default_rng(seed=abs(hash(ticker)) % (2 ** 32))
    # 模拟每日收益率
    # loc=0.0002：均值为 0.02%，表示每日预期收益率
    # scale=0.009：标准差为 0.9%，表示收益率的波动程度
    daily_returns = rng.normal(loc=0.0002, scale=0.009, size=len(date_idx))
    price_series = base_price*(1 + daily_returns).cumprod() # cumprod计算复利
    return pd.Series(price_series, index=date_idx, name=ticker)

# 多层 fallback
def download_mkt_data(maxretry: int = 5, base_sleep_sec: int = 60):
    """
    下载 SPY、AGG 和 SHY 过去 10 年的每日价格数据，并将数据保存为 CSV 文件
    首选yf, 速率限制就用stooq, 不行就用合成数据
    """
    print("-----正在下载市场数据-----")
    series_list = []

    for ticker in ["SPY", "AGG", "SHY"]:
            print(f"正在下载{ticker}...")
        # try:
        #     series = _fetch_data_from_yf(ticker, maxretry, base_sleep_sec)
        #     source = "yf"
        # except RuntimeError as e:
        #     print(f"yf不可下载{ticker}：{e},尝试stooq...")
            try:
                series = _fetch_data_from_stooq(ticker)
                source = "stooq"
            except Exception as s_e:
                print(f"stooq不可下载{ticker}：{s_e},尝试合成数据...")
                series = _gen_synthetic_series(ticker)
                source = "synthetic"
            print(f"    {ticker} 使用 {source} 数据，共 {len(series)} 行.")
            series_list.append(series)

    mkt_df = pd.concat(series_list, axis=1).sort_index()
    output_path = os.path.join(DATA_DIR, "mkt_data.csv")
    mkt_df.to_csv(output_path)
    print(f"市场数据保存在 {output_path}")
    return mkt_df

def download_user_data():
    """
    下载 银行营销数据集(id222)
    """
    print("-----正在下载用户数据-----")
    bank_m = fetch_ucirepo(id=222)
    # 提取数据集中的特征数据（输入变量）包含客户的各种属性（如年龄、职业、收入等）
    X = bank_m.data.features
    # 提取目标变量（输出变量）通常是 "是否订阅定期存款" 这样的标签
    y = bank_m.data.targets
    b_m_full_df = pd.concat([X, y], axis=1)
    output_path = os.path.join(DATA_DIR,"bm_full.csv")
    b_m_full_df.to_csv(output_path, index=False, sep=';')
    print(f"用户数据保存在 {output_path}")
    return b_m_full_df



def generate_offline_dataset():
    """
    通过在市场环境中为多个用户模拟一个基于规则的策略，来生成一个离线强化学习数据集。
    这个数据集将用于训练 CQL 算法。
    """
    print("-----开始生成离线数据集-------")

    # 1. 加载用户数据以供采样
    if not os.path.exists(USER_DATA_FILE):
        raise FileNotFoundError(f"用户数据未找到于 {USER_DATA_FILE}。请先运行下载脚本。")
    user_df = pd.read_csv(USER_DATA_FILE, sep=';')
    # 清理列名
    user_df.columns = [col.strip().replace('"', '') for col in user_df.columns]
    
    # 为模拟采样用户
    if len(user_df) < N_USERS_TO_SIMULATE:
        print(f"警告: 请求模拟 {N_USERS_TO_SIMULATE} 个用户, 但只有 {len(user_df)} 个可用。将使用所有用户。")
        sampled_users_df = user_df
    else:
        sampled_users_df = user_df.sample(n=N_USERS_TO_SIMULATE, random_state=42) # 随机抽取用户

    # 2. 初始化环境并获取状态维度
    try:
        env = MarketEnv()
        obs_dim = get_state_dim()
        print(f"obs维度: {obs_dim}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"初始化环境时出错: {e}")
        print("请确保您已成功download 市场数据，用户数据 和 train_user_model。")
        return

    # 3. 准备列表以存储所有转换
    obss, acts, rwds, dones = [], [], [], []

    # 为了解决生成500个轨迹速度慢的问题，
    # 每个用户只在进入循环前调用一次 get_accept_prob(user_profile)，
    # 并把结果随手缓存成变量 user_accept_prob；同理，allowed_cards_for_user 也在外层调用一次，
    # 而不是每个时间步都重新筛卡。
    # 循环内部就只做状态拼接、环境步进和一次 compute_reward(..., accept_prob=user_accept_prob)，
    # 省掉了大量重复的 predict_proba 和卡片筛选
    print(f"正在为 {len(sampled_users_df)} 个用户模拟轨迹...")
    for _, user_row in tqdm(sampled_users_df.iterrows(), total=len(sampled_users_df)):
        # 为当前用户创建一个经过清理的 UserProfile 对象
        user_profile = user_row_to_profile(user_row)
        user_vector = make_up_to_vec(user_profile)
        user_accept_prob = get_accept_prob(user_profile)
        allowed_cards = allowed_cards_for_user(user_profile.risk_bucket)

        # 为每个新用户重置环境
        info = env.reset()

        done = False
        while not done:
            # a. 构建当前状态
            state_vec = build_state_vec(
                mkt_features=info['mkt_sshot'],
                user_profile=user_profile,
                curr_alloc=info['curr_alloc'],
                user_vector=user_vector
            )

            # b. 获取允许的动作，并使用基于规则的策略选择一个
            action_id, _ = policy_based_rule(state_vec, allowed_cards, user_profile.risk_bucket)

            # c. 步进环境
            market_return, done, next_info = env.step(action_id)

            # d. 计算奖励
            # 为简单起见，在数据集生成阶段我们将忽略回撤
            reward = compute_reward(
                market_return,
                user_profile,
                drawdown=0.0,
                accept_prob=user_accept_prob,
            )

            # e. 存储转换
            obss.append(state_vec)
            acts.append(action_id)
            rwds.append(reward)
            dones.append(done)

            # 更新循环所使用的环境信息
            info = next_info

    print(f"\n总共生成了 {len(obss)} 个转换。")

    # 4. 创建并保存 d3rlpy MDPDataset
    obss_array = np.asarray(obss, dtype=np.float32)
    acts_array = np.asarray(acts, dtype=np.int64)
    rwds_array = np.asarray(rwds, dtype=np.float32)
    dones_array = np.asarray(dones, dtype=np.float32)

    dataset = MDPDataset(
        observations=obss_array,
        actions=acts_array,
        rewards=rwds_array,
        terminals=dones_array,
        action_space=ActionSpace.DISCRETE,
        action_size=get_act_space_size(),
    )
    
    dataset.dump(OUTPUT_DATASET_PATH)
    print(f"离线数据集已保存至 {OUTPUT_DATASET_PATH}")

if __name__ == "__main__":
    # download_mkt_data()
    generate_offline_dataset()
    print("\n离线数据集生成过程完成。")
