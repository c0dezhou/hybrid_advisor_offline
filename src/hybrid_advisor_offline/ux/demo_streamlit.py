"""
Streamlit 演示应用：Hybrid Advisor Offline
--------------------------------------------

功能概览
1. 载入最新的 CQL 模型与市场快照（缓存）。
2. 在左侧面板输入客户画像、账户规模与风险检查设定。
3. 右侧实时展示合规可用动作的 Q 值、推荐排序与合规解释。
4. 输出可复制的审计摘要，便于审计追溯。

运行方式
    python -m streamlit run hybrid_advisor_offline/ux/demo_streamlit.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from hybrid_advisor_offline.llm.text_translator import refine_text, translator_enabled
from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import (
    ALL_CARDS,
    get_card_by_id,
)
from hybrid_advisor_offline.engine.act_safety.act_filter import allowed_cards_for_user
from hybrid_advisor_offline.engine.envs.market_envs import MarketEnv
from hybrid_advisor_offline.engine.personal.personal_prior import (
    build_personal_prior,
    infer_prefs_from_profile,
)
from hybrid_advisor_offline.engine.policy.explain import build_explain_pack
from hybrid_advisor_offline.engine.state.state_builder import (
    MarketSnapshot,
    UserProfile,
    build_state_vec,
)
from hybrid_advisor_offline.offline.analysis.segment_metrics import plot_segment_bars
from hybrid_advisor_offline.offline.trainrl import train_cql
from hybrid_advisor_offline.offline.trainrl.train_cql import (
    load_cql_policy_from_paths,
)

PAGE_ICON = "🧭"
TOP_K = 3
REPORTS_DIR = Path("./reports")
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
DEMO_DATASET_PATH = os.getenv("DEMO_DATASET_PATH", "./data/offline_dataset_demo.h5")
DEMO_MODEL_PATH = os.getenv("DEMO_MODEL_PATH", "./models/cql_demo.pt")
FULL_DATASET_PATH = os.getenv(
    "STREAMLIT_DATASET_PATH",
    getattr(train_cql, "DATASET_PATH", "./data/offline_dataset.h5"),
)
FULL_MODEL_PATH = os.getenv(
    "STREAMLIT_MODEL_PATH",
    getattr(train_cql, "MODEL_SAVE_PATH", "./models/cql_discrete_model.pt"),
)


def _predict_q_values(policy, state_vec: np.ndarray) -> np.ndarray:
    action_count = len(ALL_CARDS)
    state_batch = np.repeat(state_vec[None, :], action_count, axis=0)
    action_batch = np.arange(action_count, dtype=np.int64)
    return policy.predict_value(state_batch, action_batch)


@st.cache_resource(show_spinner=False)
def load_resources(
    demo_mode: bool,
    dataset_path: str,
    model_path: str,
):
    """加载 CQL 策略与最新市场快照；失败时返回 (None, None)。"""
    try:
        policy = load_cql_policy_from_paths(
            dataset_path,
            model_path,
            require_gpu=False,
        )
        env = MarketEnv()
        latest_snapshot: MarketSnapshot = env.mkt_sshots[-1]
        return policy, latest_snapshot
    except Exception as exc:  # pragma: no cover - UI 兜底
        st.error(f"模型或数据加载失败：{exc}")
        return None, None


def _collect_user_inputs() -> Dict:
    st.sidebar.subheader("客户画像与偏好")

    age = st.sidebar.slider("年龄", min_value=20, max_value=80, value=42)
    balance = int(st.sidebar.number_input("可投资资产 (¥)", min_value=10000, max_value=5_000_000, value=500_000, step=50_000))
    job = st.sidebar.selectbox(
        "职业",
        ["management", "technician", "admin.", "services", "retired", "student", "blue-collar"],
        index=0,
    )
    marital = st.sidebar.selectbox("婚姻状况", ["single", "married", "divorced"], index=1)
    education = st.sidebar.selectbox("教育水平", ["primary", "secondary", "tertiary", "unknown"], index=2)
    housing = st.sidebar.radio("住房贷款", ["yes", "no"], index=1, horizontal=True)
    loan = st.sidebar.radio("消费贷款", ["yes", "no"], index=1, horizontal=True)
    default = st.sidebar.radio("历史违约", ["no", "yes"], index=0, horizontal=True)

    alloc_templates = {
        "稳健型 (40/40/20)": (0.4, 0.4, 0.2),
        "保守型 (20/30/50)": (0.2, 0.3, 0.5),
        "进取型 (60/30/10)": (0.6, 0.3, 0.1),
    }
    alloc_label = st.sidebar.selectbox("当前组合", list(alloc_templates.keys()), index=0)
    current_alloc = np.array(alloc_templates[alloc_label], dtype=np.float32)

    profile = UserProfile(
        age=age,
        job=job,
        marital=marital,
        education=education,
        default=default,
        balance=balance,
        housing=housing,
        loan=loan,
    )

    return {
        "profile": profile,
        "current_alloc": current_alloc,
    }


def _format_percentage_vector(vec: List[float]) -> str:
    parts = [f"{int(x * 100):02d}%" for x in vec]
    return f"股票 {parts[0]} / 债券 {parts[1]} / 现金 {parts[2]}"


def _list_report_files(pattern: str) -> List[Path]:
    if not REPORTS_DIR.exists():
        return []
    return sorted(REPORTS_DIR.glob(pattern))


def _render_segment_dashboard():
    csv_files = _list_report_files("segment_metrics_*.csv")
    if not csv_files:
        st.info("当前 reports/ 下没有 segment_metrics_*.csv，请先运行分析脚本。")
        return
    options = {f.name: f for f in csv_files}
    selected = st.selectbox("选择指标文件", list(options.keys()), key="segment_csv")
    df = pd.read_csv(options[selected])
    st.dataframe(df, use_container_width=True)
    fig = plot_segment_bars(df, output_path=None)
    st.pyplot(fig, use_container_width=True)
    fig.clf()


def _render_policy_diff_dashboard():
    json_files = _list_report_files("policy_diff_cases_*.json")
    if not json_files:
        st.info("当前 reports/ 下没有 policy_diff_cases_*.json。")
        return
    options = {f.name: f for f in json_files}
    selected = st.selectbox("选择策略差异文件", list(options.keys()), key="policy_diff_file")
    with options[selected].open("r", encoding="utf-8") as fp:
        cases = json.load(fp)
    if not cases:
        st.warning("文件为空。")
        return
    segments = sorted({case.get("user_segment", "unknown") for case in cases})
    seg_choice = st.selectbox("筛选分组", segments, key="policy_diff_segment")
    filtered = [case for case in cases if case.get("user_segment") == seg_choice]
    if not filtered:
        st.info("该分组下暂无样本。")
        return
    limit = st.slider(
        "展示样本数量",
        min_value=1,
        max_value=min(len(filtered), 20),
        value=min(5, len(filtered)),
        key="policy_diff_limit",
    )
    rows = []
    for case in filtered[:limit]:
        row = {
            "segment": case.get("user_segment"),
            "state_step": case.get("state_step"),
            "rule_card": case.get("rule", {}).get("card_id"),
            "rule_equity": case.get("rule", {}).get("equity_weight"),
            "bc_card": case.get("bc", {}).get("card_id"),
            "bcq_card": case.get("bcq", {}).get("card_id"),
        }
        rows.append(row)
    st.table(pd.DataFrame(rows))


def render_recommendations(policy, snapshot, profile: UserProfile, current_alloc: np.ndarray):
    state_vec = build_state_vec(snapshot, profile, current_alloc)
    q_values = _predict_q_values(policy, state_vec)

    allowed_cards = allowed_cards_for_user(profile.risk_bucket)
    allowed_ids = [card.act_id for card in allowed_cards]

    priors = build_personal_prior(
        allowed_ids,
        prefs=infer_prefs_from_profile(profile),
        risk_bucket=profile.risk_bucket,
    )

    mask = np.full_like(q_values, -np.inf, dtype=np.float32)
    for act_id in allowed_ids:
        if 0 <= act_id < len(mask):
            mask[act_id] = 0.0
    masked_q = q_values + mask
    if priors:
        prior_vec = np.zeros_like(q_values)
        for act_id, bump in priors.items():
            if 0 <= act_id < len(prior_vec):
                prior_vec[act_id] = bump
        masked_q = masked_q + prior_vec

    ranked_ids = sorted(allowed_ids, key=lambda aid: masked_q[aid], reverse=True)
    if not ranked_ids:
        st.warning("当前约束下没有可用的动作卡片，请调整输入。")
        return

    if translator_enabled():
        st.info("文案润色：已开启（USE_LLM_TRANSLATOR=1）", icon="✨")
    else:
        st.info("文案润色：关闭，可设置 USE_LLM_TRANSLATOR=1 启用。", icon="💬")

    st.markdown("### 推荐卡片 TOP-3")
    for idx, act_id in enumerate(ranked_ids[:TOP_K], start=1):
        card = get_card_by_id(act_id)
        explain_pack = build_explain_pack(card, profile.risk_bucket)
        explain_text, translator_meta = refine_text(
            explain_pack["customer_friendly_text"],
            {
                "card_id": card.card_id,
                "card_risk_level": card.risk_level,
                "user_risk_bucket": profile.risk_bucket,
                "target_alloc": card.target_alloc,
            },
        )
        q_score = float(masked_q[act_id])
        hash_digest = hashlib.sha256(explain_pack["audit_text"].encode("utf-8")).hexdigest()[:12]
        with st.container(border=True):
            st.write(f"**#{idx} · {card.card_id}** ｜ 目标配置 {_format_percentage_vector(card.target_alloc)}")
            cols = st.columns([1, 1, 1])
            cols[0].metric("模型 Q 值", f"{q_score:.3f}")
            cols[1].metric("策略风险", ["保守", "稳健", "进取"][card.risk_level])
            cols[2].metric("审计摘要哈希", hash_digest)
            st.caption(explain_text)
            if translator_meta not in ("translator_disabled", "translator_no_change"):
                st.caption(f"（文案润色：{translator_meta}）")

    st.markdown("---")
    st.markdown("#### 合规可用动作的 Q 值分布")
    df = pd.DataFrame(
        {
            "card_id": [get_card_by_id(aid).card_id for aid in allowed_ids],
            "q_value": [float(masked_q[aid]) for aid in allowed_ids],
        }
    ).sort_values("q_value", ascending=False)
    st.bar_chart(df, x="card_id", y="q_value", color="#4B8BBE")

    with st.expander("原始客户画像 / 状态向量特征"):
        st.json({"profile": asdict(profile), "current_alloc": current_alloc.tolist()})


def render_analysis_tab():
    st.subheader("分群指标（CSV）")
    _render_segment_dashboard()
    st.markdown("---")
    st.subheader("策略差异样本")
    _render_policy_diff_dashboard()


def main():
    st.set_page_config(
        page_title="Hybrid Advisor Offline Demo",
        page_icon=PAGE_ICON,
        layout="wide",
    )
    st.title("Hybrid Advisor Offline · 前端演示")
    st.caption(
        "离线 CQL 模型 + 合规安全壳。输入客户画像后即可查看推荐卡片、Q 值与审计摘要。"
    )

    active_dataset = DEMO_DATASET_PATH if DEMO_MODE else FULL_DATASET_PATH
    active_model = DEMO_MODEL_PATH if DEMO_MODE else FULL_MODEL_PATH
    policy, snapshot = load_resources(DEMO_MODE, active_dataset, active_model)
    if policy is None or snapshot is None:
        st.stop()

    inputs = _collect_user_inputs()

    tab_reco, tab_analysis = st.tabs(["实时推荐", "千人千面分析"])
    with tab_reco:
        col_left, col_right = st.columns([0.35, 0.65], gap="large")
        with col_left:
            st.subheader("安全壳&状态总览")
            if DEMO_MODE:
                st.info(
                    "Demo 模式：使用轻量小模型，仅用于快速展示。",
                    icon="⚡",
                )
            st.metric("风险等级 (0=保守,2=进取)", inputs["profile"].risk_bucket)
            st.metric(
                "当前配置",
                _format_percentage_vector(inputs["current_alloc"]),
                help="用于拼接状态向量，也可作为组合调仓参考。",
            )
        with col_right:
            render_recommendations(
                policy,
                snapshot,
                profile=inputs["profile"],
                current_alloc=inputs["current_alloc"],
            )

    with tab_analysis:
        render_analysis_tab()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "✅ 当前界面仅用于演示，不会触发真实交易。\n\n"
        "☑️ 当 CQL 模型或数据未准备好时，应用会提示错误。"
    )


if __name__ == "__main__":
    main()
