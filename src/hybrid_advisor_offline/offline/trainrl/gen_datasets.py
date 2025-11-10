# fetch data
import argparse
import io
import os
import sys
import time
import shutil
import tempfile
import subprocess
from dataclasses import asdict
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ucimlrepo import fetch_ucirepo

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
OUTPUT_DATASET_PATH = os.environ.get("OUTPUT_DATASET_PATH", "./data/offline_dataset.h5")  # 输出的数据集文件
BEHAVIOR_META_SUFFIX = "_behavior.npz"
EPISODE_SUMMARY_SUFFIX = "_episodes.csv"
# N_USERS_TO_SIMULATE = 500  # 用于模拟的用户数量,生成500条轨迹
# N_USERS_TO_SIMULATE = 1000  # 用于模拟的用户数量,生成1000条轨迹
# N_USERS_TO_SIMULATE = 20000  # 用于模拟的用户数量,生成10000条轨迹
N_USERS_TO_SIMULATE = 45000  # 用于模拟的用户数量,生成10000条轨迹
EPISODE_MAX_STEPS = int(os.getenv("EPISODE_MAX_STEPS", "252"))  # 默认每条轨迹最长252步（一年交易日）
POLICY_EPS_FIXED = float(os.getenv("POLICY_EPSILON", "0.2"))  # 固定 ε 值，让 20% 的时间随机动作
POLICY_SEED = 42
EXPERIMENT_MODE = os.getenv("EXPERIMENT_MODE", "full").lower()
_FAST_MODE_NAMES = {"fast", "dev"}
IS_FAST_MODE = EXPERIMENT_MODE in _FAST_MODE_NAMES
FAST_MODE_USER_CAP = int(os.getenv("FAST_MODE_USER_CAP", "2000"))
FAST_MODE_STEP_CAP = int(os.getenv("FAST_MODE_STEP_CAP", "64"))

if IS_FAST_MODE:
    N_USERS_TO_SIMULATE = min(N_USERS_TO_SIMULATE, FAST_MODE_USER_CAP)
    EPISODE_MAX_STEPS = min(EPISODE_MAX_STEPS, FAST_MODE_STEP_CAP)

FILTER_LOW_RETURN = os.getenv("FILTER_LOW_RETURN", "0") == "1"
DATASET_KEEP_TOP_FRAC = float(os.getenv("DATASET_KEEP_TOP_FRAC", "1.0"))  # 保留回报 top 比例
_min_return_env = os.getenv("DATASET_MIN_RETURN", "").strip()
DATASET_MIN_RETURN = float(_min_return_env) if _min_return_env else None  # 默认不过滤


def _profile_to_meta(profile: UserProfile) -> dict:
    """将 UserProfile 转换为纯 dict，方便写入 npz。"""
    return asdict(profile)


def _behavior_meta_path(dataset_path: str = OUTPUT_DATASET_PATH) -> str:
    """根据数据集路径推导行为策略统计文件路径。"""
    base, _ = os.path.splitext(dataset_path)
    return f"{base}{BEHAVIOR_META_SUFFIX}"


def _episode_summary_path(dataset_path: str = OUTPUT_DATASET_PATH) -> str:
    base, _ = os.path.splitext(dataset_path)
    return f"{base}{EPISODE_SUMMARY_SUFFIX}"

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
    print(
        f"[gen_datasets] mode={EXPERIMENT_MODE}, fast_mode={IS_FAST_MODE}, "
        f"users={N_USERS_TO_SIMULATE}, max_steps={EPISODE_MAX_STEPS}"
    )

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
        # 随机不放回抽取用户
        sampled_users_df = user_df.sample(n=N_USERS_TO_SIMULATE, random_state=42,replace=False)
        

    # 2. 初始化环境并获取状态维度
    try:
        env = MarketEnv(max_episode_steps=EPISODE_MAX_STEPS)
        obs_dim = get_state_dim()
        print(f"环境初始化成功，单条轨迹最大步数: {EPISODE_MAX_STEPS}")
        print(f"obs维度: {obs_dim}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"初始化环境时出错: {e}")
        print("请确保您已成功download 市场数据，用户数据 和 train_user_model。")
        return

    total_users = len(sampled_users_df)
    max_steps = total_users * EPISODE_MAX_STEPS
    if max_steps <= 0:
        print("没有可生成的用户或步数，直接返回。")
        return

    tmp_mm_dir = tempfile.mkdtemp(prefix="offline_dataset_memmap_")
    mm_handles: list[np.memmap] = []

    def _alloc_mm(name: str, shape, dtype):
        """轻量封装一下 np.memmap，省得每次写重复代码。"""
        path = os.path.join(tmp_mm_dir, f"{name}.dat")
        mmap = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
        mm_handles.append(mmap)
        return mmap

    obss_mem = _alloc_mm("obs", (max_steps, obs_dim), np.float32)
    acts_mem = _alloc_mm("acts", (max_steps,), np.int64)
    rwds_mem = _alloc_mm("rews", (max_steps,), np.float32)
    dones_mem = _alloc_mm("dones", (max_steps,), np.float32)
    prop_mem = _alloc_mm("props", (max_steps,), np.float32)
    episode_ids_mem = _alloc_mm("episode_ids", (max_steps,), np.int32)

    rng = np.random.default_rng(POLICY_SEED)
    ep_id_counter = 0
    ep_return_bucket: list[float] = []
    ep_length_bucket: list[int] = []
    episode_profiles: list[dict] = []
    row_ptr = 0

    print(f"正在为 {len(sampled_users_df)} 个用户模拟轨迹...")
    # 主循环：逐个用户跑一条 episode，把 (s,a,r,p) 写入 memmap
    for _, user_row in tqdm(sampled_users_df.iterrows(), total=len(sampled_users_df)):
        user_profile = user_row_to_profile(user_row)
        user_vector = make_up_to_vec(user_profile)
        user_accept_prob = get_accept_prob(user_profile)
        allowed_cards = allowed_cards_for_user(user_profile.risk_bucket)

        info = env.reset()
        ep_return = 0.0
        ep_steps = 0

        done = False
        while not done:
            state_vec = build_state_vec(
                mkt_features=info['mkt_sshot'],
                user_profile=user_profile,
                curr_alloc=info['curr_alloc'],
                user_vector=user_vector
            )

            action_id, _, propensity = policy_based_rule(
                state_vec,
                allowed_cards,
                user_profile.risk_bucket,
                exploration_rate=POLICY_EPS_FIXED,
                rng=rng,
                return_propensity=True,
            )

            market_return, done, next_info = env.step(action_id)
            drawdown = float(next_info.get("drawdown", 0.0))

            reward = compute_reward(
                market_return,
                drawdown=drawdown,
                user_profile=user_profile,
                accept_prob=user_accept_prob,
            )
            ep_return += reward
            ep_steps += 1

            if row_ptr >= max_steps:
                raise RuntimeError(
                    "预估的 max_steps 太小，写入越界。请调整 EPISODE_MAX_STEPS 或减少用户数量。"
                )
            obss_mem[row_ptr] = state_vec
            acts_mem[row_ptr] = action_id
            rwds_mem[row_ptr] = reward
            dones_mem[row_ptr] = 1.0 if done else 0.0
            prop_mem[row_ptr] = float(propensity)
            episode_ids_mem[row_ptr] = ep_id_counter
            row_ptr += 1

            info = next_info

        episode_profiles.append(_profile_to_meta(user_profile))
        ep_id_counter += 1
        ep_return_bucket.append(ep_return)
        ep_length_bucket.append(ep_steps)

    print(f"\n总共生成了 {row_ptr} 个转换。")

    # 3.1 轨迹质量过滤（可选）
    def _filter_by_returns(e_returns, keep_frac, min_return):
        if not FILTER_LOW_RETURN:
            return None
        keep_frac = max(0.0, min(1.0, keep_frac))
        min_return = None if min_return is None else float(min_return)
        if keep_frac >= 0.999 and (min_return is None):
            return None
        sorted_eps = sorted(enumerate(e_returns), key=lambda kv: kv[1], reverse=True)
        if not sorted_eps:
            return None
        keep_count = max(1, int(len(sorted_eps) * keep_frac))
        keep_ids = {ep_id for ep_id, _ in sorted_eps[:keep_count]}
        if min_return is not None:
            for ep_id, total in sorted_eps:
                if total >= min_return:
                    keep_ids.add(ep_id)
        if len(keep_ids) == len(e_returns):
            return None
        return keep_ids

    keep_episode_ids = _filter_by_returns(ep_return_bucket, DATASET_KEEP_TOP_FRAC, DATASET_MIN_RETURN)
    if keep_episode_ids:
        keep_flags = np.isin(np.arange(len(ep_return_bucket)), list(keep_episode_ids))
        ep_return_bucket = [ret for ret, keep in zip(ep_return_bucket, keep_flags) if keep]
        ep_length_bucket = [length for length, keep in zip(ep_length_bucket, keep_flags) if keep]
        episode_profiles = [profile for profile, keep in zip(episode_profiles, keep_flags) if keep]

    # 4. 创建并保存 d3rlpy MDPDataset
    used_slice = slice(0, row_ptr)
    obss_array = obss_mem[used_slice]
    acts_array = acts_mem[used_slice]
    rwds_array = rwds_mem[used_slice]
    dones_array = dones_mem[used_slice]
    prop_array = prop_mem[used_slice]
    episode_ids_array = episode_ids_mem[used_slice]

    if keep_episode_ids:
        mask = np.isin(episode_ids_array, list(keep_episode_ids))
        print(f"过滤低质量轨迹：保留 {mask.sum()} / {len(mask)} 步（{mask.sum()/len(mask):.2%}）。")
        obss_array = obss_array[mask]
        acts_array = acts_array[mask]
        rwds_array = rwds_array[mask]
        dones_array = dones_array[mask]
        prop_array = prop_array[mask]
        episode_ids_array = episode_ids_array[mask]

    # 最终把内存片段打包成 d3rlpy 的 MDPDataset，方便后续 BC / BCQ / CQL 复用
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

    behavior_meta_path = _behavior_meta_path(OUTPUT_DATASET_PATH)
    np.savez(
        behavior_meta_path,
        propensities=prop_array,
        actions=acts_array,
        episode_ids=episode_ids_array,
        terminals=dones_array,
        epsilon=np.array([POLICY_EPS_FIXED], dtype=np.float32),
        user_profiles=np.array(episode_profiles, dtype=object),
    )
    print(f"行为策略倾向已保存至 {behavior_meta_path}")

    summary_df = pd.DataFrame(
        {
            "episode_id": np.arange(len(ep_return_bucket), dtype=np.int32),
            "total_reward": ep_return_bucket,
            "length": ep_length_bucket,
        }
    )
    summary_path = _episode_summary_path(OUTPUT_DATASET_PATH)
    summary_df.to_csv(summary_path, index=False)
    print(f"episode 汇总已保存至 {summary_path}")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="生成离线数据集，可选地触发评估管线。",
    )
    parser.add_argument(
        "--re-eval",
        action="store_true",
        help="生成数据后立即运行 eval_policy 对最新数据/模型做一次评估。",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default="./models/cql_discrete_model.pt",
        help="评估所使用的模型路径（默认: ./models/cql_discrete_model.pt）。",
    )
    parser.add_argument(
        "--extra-eval-args",
        nargs=argparse.REMAINDER,
        help="传递给评估脚本的附加参数（放在 --extra-eval-args 之后）。",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help=f"rule-based 行为策略的 ε 探索率（默认 {POLICY_EPS_FIXED}）。",
    )
    parser.add_argument(
        "--filter-low-return",
        action="store_true",
        help="启用低回报轨迹过滤（默认关闭，调试用）。",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="输出数据集路径（默认读取 OUTPUT_DATASET_PATH 环境变量或 ./data/offline_dataset.h5）。",
    )
    parser.add_argument(
        "--num-users",
        type=int,
        default=None,
        help="生成轨迹的用户数量（默认 N_USERS_TO_SIMULATE）。",
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=None,
        help="单条轨迹的最大步数（默认 EPISODE_MAX_STEPS）。",
    )
    parser.add_argument(
        "--top-frac",
        type=float,
        default=None,
        help="当启用过滤时，保留回报 Top 百分比（默认读取环境变量 DATASET_KEEP_TOP_FRAC）。",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=None,
        help="当启用过滤时，保留回报>=该阈值的 episode（默认读取 DATASET_MIN_RETURN）。",
    )
    return parser.parse_args()


def _run_evaluation(model_path: str, extra_args: list[str] | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "hybrid_advisor_offline.offline.eval.eval_policy",
        "--dataset",
        OUTPUT_DATASET_PATH,
        "--model",
        model_path,
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("\n--- 重新运行评估 ---")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = _parse_args()

    global POLICY_EPS_FIXED, FILTER_LOW_RETURN, DATASET_KEEP_TOP_FRAC, DATASET_MIN_RETURN
    global OUTPUT_DATASET_PATH, N_USERS_TO_SIMULATE, EPISODE_MAX_STEPS
    if args.epsilon is not None:
        POLICY_EPS_FIXED = max(0.0, min(1.0, args.epsilon))
        print(f"使用自定义 ε 探索率: {POLICY_EPS_FIXED}")
    if args.filter_low_return:
        FILTER_LOW_RETURN = True
        if args.top_frac is not None:
            DATASET_KEEP_TOP_FRAC = args.top_frac
        if args.min_return is not None:
            DATASET_MIN_RETURN = args.min_return
        print(
            f"已启用轨迹过滤：top_frac={DATASET_KEEP_TOP_FRAC}, "
            f"min_return={DATASET_MIN_RETURN}"
        )
    if args.dataset_path:
        OUTPUT_DATASET_PATH = args.dataset_path
        print(f"使用自定义数据集路径: {OUTPUT_DATASET_PATH}")
    if args.num_users is not None:
        N_USERS_TO_SIMULATE = max(1, args.num_users)
        print(f"使用自定义用户数量: {N_USERS_TO_SIMULATE}")
    if args.episode_steps is not None:
        EPISODE_MAX_STEPS = max(1, args.episode_steps)
        print(f"使用自定义 episode 步数: {EPISODE_MAX_STEPS}")
    if IS_FAST_MODE:
        N_USERS_TO_SIMULATE = min(N_USERS_TO_SIMULATE, FAST_MODE_USER_CAP)
        EPISODE_MAX_STEPS = min(EPISODE_MAX_STEPS, FAST_MODE_STEP_CAP)
        print(
            f"[gen_datasets] fast_dev 限制生效 -> users={N_USERS_TO_SIMULATE}, "
            f"max_steps={EPISODE_MAX_STEPS}"
        )

    # download_mkt_data()
    generate_offline_dataset()
    print("\n离线数据集生成过程完成。")

    if args.re_eval:
        _run_evaluation(args.eval_model, args.extra_eval_args)


if __name__ == "__main__":
    main()

# USE_CARD_FACTORY=1 \
# POLICY_EPSILON=0.4 \
# DATASET_KEEP_TOP_FRAC=0.7 \
# DATASET_MIN_RETURN=10 \
# python -m hybrid_advisor_offline.offline.trainrl.gen_datasets
