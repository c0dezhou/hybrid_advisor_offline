# fetch data
import io
import os
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ucimlrepo import fetch_ucirepo

DATA_DIR = "./data"
# SPY：标普 500 指数 ETF
# AGG：美国综合债券 ETF
# SHY：短期国债 ETF
_STOOQ_MAP = {"SPY": "spy.us", "AGG": "agg.us", "SHY": "shy.us"}

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
def download_market_data(maxretry: int = 5, base_sleep_sec: int = 60):
    """
    下载 SPY、AGG 和 SHY 过去 10 年的每日价格数据，并将数据保存为 CSV 文件
    首选yf, 速率限制就用stooq, 不行就用合成数据
    """
    print("-----正在下载市场数据-----")
    series_list = []

    for ticker in ["SPY", "AGG", "SHY"]:
        print(f"正在下载{ticker}...")
        try:
            series = _fetch_data_from_yf(ticker, maxretry, base_sleep_sec)
            source = "yf"
        except RuntimeError as e:
            print(f"yf不可下载{ticker}：{e},尝试stooq...")
            try:
                series = _fetch_data_from_stooq(ticker)
                source = "stooq"
            except Exception as s_e:
                print(f"stooq不可下载{ticker}：{s_e},尝试合成数据...")
                series = _gen_synthetic_series(ticker)
                source = "synthetic"
            print(f"    {ticker} 从 {source} 下载成功, {len(series)} 行.")
            series_list.append(series)

        market_df = pd.concat(series_list, axis=1).sort_index()
        output_path = os.path.join(DATA_DIR, "market_data.csv")
        market_df.to_csv(output_path)
        print(f"市场数据保存在 {output_path}")
        return market_df

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

def main():
    """构造数据集"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"创建文件夹: {DATA_DIR}")
    download_market_data()
    print("+"*30)
    download_user_data()
    print("\n数据集构造完毕")

if __name__ == "__main__":
    main()
