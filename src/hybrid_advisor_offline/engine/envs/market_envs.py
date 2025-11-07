import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_card_by_id
from hybrid_advisor_offline.engine.state.state_builder import MarketSnapshot

DATA_FILE = "./data/mkt_data.csv"
INITIAL_BALANCE = 1000000.0  # 初始投资组合价值
TRANSACTION_COST_PCT = 0.001  # 0.1% 的交易成本,核心作用是对过于频繁的交易施加惩罚

class MarketEnv:
    """
    模拟投资组合的模拟市场环境
    设计独立于rl的state和reward，仅专注于市场本身

    合规性：再平衡操作的设计必须通过合规部门的预先批准
    """
    def __init__(self, *, max_episode_steps: int | None = None):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"市场数据未找到：{DATA_FILE}"
                "请download_data"
            )
        # 加载、准备数据集
        self.mkt_data = pd.read_csv(DATA_FILE, index_col='Date',parse_dates=True)
        # 计算当前行与前一行的变化率，公式为 (当前值 - 上一值) / 上一值，即每日收益率
        self.daily_returns = self.mkt_data.pct_change().fillna(0)# 第一天NaN填充为0
        self.n_steps = len(self.mkt_data) #交易日总数，也就是模拟的总步数
        self.max_episode_steps = max_episode_steps or self.n_steps
    
        # 为每个时间步预先计算市场快照
        self._pre_mkt_sshots()

        # env
        self.curr_step = 0
        self.steps_in_episode = 0
        self.portfolio_value = INITIAL_BALANCE
        self.curr_alloc = np.array([0.0,0.0,1.0]) # 初始全现金

    def _pre_mkt_sshots(self):
        """
        预先计算市场快照,
        提前计算好每一天对应的市场快照（包含30天滚动收益率、波动率等）
        """
        print("为env预计算市场快照")
        self.mkt_sshots = []
        # 滚动计算的窗口
        window = 30
        # 定义年化因子：波动率（标准差）与时间的平方根成正比
        annual_factor = np.sqrt(252) # 252是一年中的交易日数量
        rolling_returens = self.mkt_data.pct_change().rolling(window=window).sum()
        rolling_vols = self.daily_returns.rolling(window=window).std()*annual_factor
        for i in range(self.n_steps):
            if i <window:
                # 数据不够30个，用0代替
                sshot = MarketSnapshot(
                    rolling_30d_returen=np.zeros(3),
                    rolling_30d_vol=np.zeros(3),
                    vix=0.0
                )
            else:
                # 使用 SPY（标普500的ETF）的历史年化波动率作为VIX指数的一个近似替代
                vix_approx = rolling_vols.iloc[i]['SPY'] 
                sshot = MarketSnapshot(
                    rolling_30d_returen=rolling_returens.iloc[i].values,
                    rolling_30d_vol=rolling_vols.iloc[i].values,
                    vix=vix_approx
                )
            self.mkt_sshots.append(sshot)

    def reset(self):
        """重置环境到初始态，在一个episode开始被调用"""
        self.curr_step = 0
        self.steps_in_episode = 0
        self.portfolio_value = INITIAL_BALANCE
        self.curr_alloc = np.array([0.0, 0.0, 1.0])

        info = {
            "mkt_sshot": self.mkt_sshots[self.curr_step],
            "curr_alloc": self.curr_alloc,
            "portfolio_value": self.portfolio_value,
        }
        return info
    
    def step(self, act_id):
        """
        将环境推进一个时间步
        """
        # 1. 从act卡片中获得目标配置
        card = get_card_by_id(act_id)
        target_alloc = np.array(card.target_alloc)

        # 2. 计算并使用交易成本
        rebalance_amount = np.sum(np.abs(target_alloc - self.curr_alloc)) # 从当前配置调整到目标配置，进行买卖操作
        cost = self.portfolio_value * rebalance_amount * TRANSACTION_COST_PCT
        self.portfolio_value -= cost # 从总资产中扣除这笔交易成本

        # 3. 更新配置
        self.curr_alloc = target_alloc

        # 4.计算当天的投资组合回报
        today_returns = self.daily_returns.iloc[self.curr_step].values
        # 今日操作总回报率，用点积
        portfolio_daily_return = np.dot(self.curr_alloc, today_returns)

        #5. 根据市场变动更新投资组合价值
        self.portfolio_value *= (1+portfolio_daily_return)

        #6. 移动到St+1
        self.curr_step += 1
        self.steps_in_episode += 1
        done = (
            self.curr_step >= self.n_steps
            or self.steps_in_episode >= self.max_episode_steps
        )
        # [ERROR] IndexError 是因为 MarketEnv.step 在推进到最后一个交易日后，
        # 把 curr_step 加到了 n_steps，但仍然试图用这个索引访问 self.mkt_sshots[self.curr_step]。
        # 列表长度是 n_steps，合法索引只有 0 … n_steps-1，所以一到结尾就越界了。
        next_idx = min(self.curr_step, self.n_steps - 1)

        # 7.准备下一个state的信息
        next_info = {
            "mkt_sshot": self.mkt_sshots[next_idx],
            "curr_alloc": self.curr_alloc,
            "portfolio_value": self.portfolio_value,
        }

        return portfolio_daily_return, done, next_info
    
if __name__ == '__main__':
    print("\n----------测试市场环境----------")
    env = MarketEnv()
    info = env.reset()
    
    print(f"初始投资组合价值: ${info['portfolio_value']:.2f}")
    print(f"初始配置: {info['curr_alloc']}")
    
    # 使用一个固定的动作（例如，均衡型投资组合）模拟几个步骤
    action_id_to_test = 2 # MOD_BALANCED
    
    for i in range(5):
        mkt_return, done, info = env.step(action_id_to_test)
        print(
            f"步骤 {env.curr_step}: "
            f"回报率={mkt_return:.4f}, "
            f"新价值=${info['portfolio_value']:.2f}, "
            f"结束={done}"
        )
    
    print("\n环境模拟测试完成。")


