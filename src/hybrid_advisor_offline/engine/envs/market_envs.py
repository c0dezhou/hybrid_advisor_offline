import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_card_by_id
from hybrid_advisor_offline.engine.state.state_builder import MarketSnapshot

_PRECOMPUTE_LOGGED = False

DEFAULT_ALLOC = np.array([0.0, 0.0, 1.0], dtype=np.float32)

DATA_FILE = "./data/mkt_data.csv"
INITIAL_BALANCE = 1000000.0  # 初始投资组合价值
TRANSACTION_COST_PCT = 0.001  # 0.1% 的交易成本,核心作用是对过于频繁的交易施加惩罚

logger = logging.getLogger(__name__)

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
        self._num_days = self.n_steps
        self.max_episode_steps = max_episode_steps or self.n_steps
        self._max_episode_steps = self.max_episode_steps
    
        # 为每个时间步预先计算市场快照
        self._pre_mkt_sshots()

        # env
        self.curr_step = 0
        self.steps_in_episode = 0
        self.portfolio_value = INITIAL_BALANCE
        self.curr_alloc = np.array([0.0,0.0,1.0]) # 初始全现金
        self.equity = 1.0
        self.high_water = 1.0
        self.curr_drawdown = 0.0
        self._has_custom_alloc = False
        self._rng, self._rng_seed = self._load_rng()
        self._start_mode = self._load_start_mode()
        self._start_block_size = self._load_start_block_size()
        self._start_pool_mode = self._load_start_pool_mode()
        self._fixed_start_idx = self._load_fixed_start_idx()
        self._embargo_days = self._load_embargo_days()
        self._start_range_raw = self._load_start_range_raw()
        self._start_idx_pool = self._build_start_idx_pool()
        self._current_start_idx = self._fixed_start_idx
        self._cursor = self._current_start_idx
        self._reset_counter = 0
        self._start_config_logged = False

    def _pre_mkt_sshots(self):
        """
        预先计算市场快照,
        提前计算好每一天对应的市场快照（包含30天滚动收益率、波动率等）
        """
        global _PRECOMPUTE_LOGGED
        if not _PRECOMPUTE_LOGGED:
            print("为env预计算市场快照")
            _PRECOMPUTE_LOGGED = True
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

    def _load_start_mode(self) -> str:
        mode = os.getenv("EPISODE_START_MODE", "").strip().lower() or "random"
        if mode not in {"fixed", "random"}:
            logger.warning(
                "EPISODE_START_MODE=%s 不受支持，已回退至 fixed。", mode
            )
            mode = "fixed"
        return mode

    def _load_start_block_size(self) -> int:
        raw = os.getenv("EPISODE_START_BLOCK_SIZE")
        if raw is None:
            return 1
        try:
            size = int(raw)
        except ValueError:
            logger.warning(
                "无法解析 EPISODE_START_BLOCK_SIZE=%s，已回退到 1。",
                raw,
            )
            return 1
        return max(1, size)

    def _load_start_pool_mode(self) -> str:
        mode = os.getenv("EPISODE_START_POOL", "").strip().lower() or "all"
        if mode not in {"all", "even", "odd"}:
            logger.warning(
                "EPISODE_START_POOL=%s 不受支持，已回退至 all。", mode
            )
            mode = "all"
        return mode

    def _load_embargo_days(self) -> int:
        raw = os.getenv("EPISODE_EMBARGO_DAYS")
        if raw is None:
            return 0
        try:
            days = int(raw)
        except ValueError:
            logger.warning(
                "无法解析 EPISODE_EMBARGO_DAYS=%s，已回退到 0。", raw
            )
            return 0
        return max(0, days)

    def _load_start_range_raw(self) -> tuple[int | None, int | None]:
        raw = os.getenv("EPISODE_START_RANGE", "").strip()
        if not raw:
            return (None, None)
        sep = ":" if ":" in raw else ","
        parts = [p.strip() for p in raw.split(sep, 1)]
        try:
            start = int(parts[0]) if parts[0] else 0
        except ValueError:
            logger.warning(
                "无法解析 EPISODE_START_RANGE=%s，已忽略。", raw
            )
            return (None, None)
        end = None
        if len(parts) > 1 and parts[1]:
            try:
                end = int(parts[1])
            except ValueError:
                logger.warning(
                    "无法解析 EPISODE_START_RANGE 末尾=%s，已忽略上限。", parts[1]
                )
        start = max(0, start)
        end = None if end is None else max(end, start)
        return (start, end)

    def _resolve_start_range(self, max_start: int) -> tuple[int | None, int | None]:
        start, end = self._start_range_raw
        if start is not None:
            start = min(start, max_start)
        if end is not None:
            end = min(end, max_start)
            if start is not None and end < start:
                end = start
        return (start, end)

    def _build_start_idx_pool(self) -> list[int]:
        if self._start_mode != "random":
            return []
        max_idx = self._max_random_start_idx()
        max_idx = max(0, max_idx)
        candidates = list(range(0, max_idx + 1))
        if not candidates:
            return [0]
        if self._start_pool_mode == "all":
            return candidates
        target_parity = 0 if self._start_pool_mode == "even" else 1
        block_size = self._start_block_size
        block_size = max(1, block_size)
        width = self._max_episode_steps + self._embargo_days
        allowed: list[int] = []
        max_start = max_idx + 1
        block_id = 0
        for block_start in range(0, max_start, block_size):
            block_end = min(block_start + block_size, max_start)
            effective_span = block_end - block_start
            if effective_span < self._max_episode_steps:
                block_id += 1
                continue
            if (block_id % 2) == target_parity:
                allowed.extend(range(block_start, block_end))
            block_id += 1
        range_min, range_max = self._resolve_start_range(max_start)
        if range_min is not None or range_max is not None:
            filtered: list[int] = []
            for idx in allowed:
                if range_min is not None and idx < range_min:
                    continue
                if range_max is not None and idx >= range_max:
                    continue
                filtered.append(idx)
            if filtered:
                allowed = filtered
            else:
                logger.warning(
                    "EPISODE_START_RANGE=%s 过滤后候选集合为空，回退到 all。",
                    os.getenv("EPISODE_START_RANGE", ""),
                )
                return candidates
        if not allowed:
            logger.warning(
                "start_pool_mode=%s block_size=%d 导致空的起点集合，已回退到 all。",
                self._start_pool_mode,
                block_size,
            )
            return candidates
        return allowed

    def _load_rng(self) -> tuple[np.random.Generator, int]:
        raw = os.getenv("EPISODE_START_SEED")
        default_seed = 42
        if raw is None:
            return np.random.default_rng(default_seed), default_seed
        try:
            seed = int(raw)
        except ValueError:
            logger.warning(
                "无法解析 EPISODE_START_SEED=%s，已回退到默认种子 %d。",
                raw,
                default_seed,
            )
            seed = default_seed
        return np.random.default_rng(seed), seed

    def _load_fixed_start_idx(self) -> int:
        raw = os.getenv("EPISODE_FIXED_START")
        if raw is None:
            return 0
        try:
            idx = int(raw)
        except ValueError:
            logger.warning(
                "无法解析 EPISODE_FIXED_START=%s，已回退到 0。", raw
            )
            idx = 0
        return max(0, min(idx, self._num_days - 1))

    def _max_random_start_idx(self) -> int:
        return max(0, self._num_days - self._max_episode_steps)

    def _sample_episode_start_idx(self) -> int:
        if self._start_mode == "fixed":
            return self._fixed_start_idx
        if self._start_idx_pool:
            idx = int(self._rng.choice(self._start_idx_pool))
            return idx
        max_idx = self._max_random_start_idx()
        if max_idx <= 0:
            return 0
        return int(self._rng.integers(0, max_idx + 1))

    def _log_start_config_once(self):
        if self._start_config_logged:
            return
        logger.info(
            (
                "MarketEnv start mode=%s, fixed_start=%d, random_range=[%d, %d], "
                "num_days=%d, max_episode_steps=%d, start_seed=%d, pool_mode=%s, "
                "block_size=%d, embargo_days=%d, pool_count=%d"
            ),
            self._start_mode,
            self._fixed_start_idx,
            0,
            self._max_random_start_idx(),
            self._num_days,
            self._max_episode_steps,
            self._rng_seed,
            self._start_pool_mode,
            self._start_block_size,
            self._embargo_days,
            len(self._start_idx_pool) if self._start_idx_pool else 0,
        )
        self._start_config_logged = True

    def reset(self):
        """重置环境到初始态，在一个episode开始被调用"""
        self._log_start_config_once()
        self._current_start_idx = self._sample_episode_start_idx()
        if self._start_mode == "random" and self._reset_counter < 5:
            logger.debug(
                "MarketEnv episode %d start idx=%d",
                self._reset_counter,
                self._current_start_idx,
            )
        self._reset_counter += 1
        self._cursor = self._current_start_idx
        self.curr_step = self._cursor
        self.steps_in_episode = 0
        self.portfolio_value = INITIAL_BALANCE
        self.curr_alloc = np.array([0.0, 0.0, 1.0])
        self.equity = 1.0
        self.high_water = 1.0
        self.curr_drawdown = 0.0
        self._has_custom_alloc = False

        info = {
            "mkt_sshot": self.mkt_sshots[self.curr_step],
            "curr_alloc": self.curr_alloc,
            "portfolio_value": self.portfolio_value,
            "drawdown": self.curr_drawdown,
            "episode_start_idx": self._current_start_idx,
            "has_custom_alloc": self._has_custom_alloc,
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
        self._has_custom_alloc = not np.allclose(
            self.curr_alloc, np.array([0.0, 0.0, 1.0]), atol=1e-5
        )

        # 4.计算当天的投资组合回报
        today_returns = self.daily_returns.iloc[self._cursor].values
        # 今日操作总回报率，用点积
        portfolio_daily_return = np.dot(self.curr_alloc, today_returns)

        #5. 根据市场变动更新投资组合价值
        self.portfolio_value *= (1+portfolio_daily_return)
        # 维护净值曲线与回撤，供 reward 使用
        self.equity *= (1.0 + portfolio_daily_return)
        self.high_water = max(self.high_water, self.equity)
        if self.high_water > 0:
            self.curr_drawdown = max(0.0, 1.0 - self.equity / self.high_water)
        else:
            self.curr_drawdown = 0.0

        #6. 移动到St+1
        self._cursor += 1
        self.curr_step = self._cursor
        self.steps_in_episode += 1
        done = (
            self._cursor >= self._num_days
            or self.steps_in_episode >= self._max_episode_steps
        )
        # [ERROR] IndexError 是因为 MarketEnv.step 在推进到最后一个交易日后，
        # 把 curr_step 加到了 n_steps，但仍然试图用这个索引访问 self.mkt_sshots[self.curr_step]。
        # 列表长度是 n_steps，合法索引只有 0 … n_steps-1，所以一到结尾就越界了。
        next_idx = min(self._cursor, self._num_days - 1)

        # 7.准备下一个state的信息
        next_info = {
            "mkt_sshot": self.mkt_sshots[next_idx],
            "curr_alloc": self.curr_alloc,
            "portfolio_value": self.portfolio_value,
            "drawdown": self.curr_drawdown,
            "episode_start_idx": self._current_start_idx,
            "has_custom_alloc": self._has_custom_alloc,
        }

        return portfolio_daily_return, done, next_info

    def set_custom_alloc(self, alloc: np.ndarray) -> None:
        """允许业务层在 episode 开始前填入用户的当前仓位。"""
        alloc = np.asarray(alloc, dtype=np.float32)
        if not np.isclose(alloc.sum(), 1.0, atol=1e-3):
            raise ValueError("alloc must sum to 1.0")
        self.curr_alloc = alloc
        self._has_custom_alloc = not np.allclose(alloc, np.array([0.0, 0.0, 1.0]), atol=1e-5)
    
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
