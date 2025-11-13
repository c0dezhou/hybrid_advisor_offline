"""
Streamlit 演示应用：Hybrid Advisor Offline
--------------------------------------------

功能概览
1. 载入最新的策略模型（可切换 BC / BCQ / CQL）与市场快照。
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
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

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
from hybrid_advisor_offline.offline.eval.policy_loader import load_policy_artifact

PAGE_ICON = "🧭"
TOP_K = 3
REPORTS_DIR = Path("./reports")
MODEL_REGISTRY = {
    "bcq": {"label": "BCQ（默认）", "path": Path("./models/bcq_reward_personal.pt")},
    "bc": {"label": "BC", "path": Path("./models/bc_reward_personal.pt")},
    "cql": {"label": "CQL", "path": Path("./models/cql_reward_personal.pt")},
}
MODEL_ORDER = ["bcq", "bc", "cql"]
DEFAULT_MODEL_KEY = "bcq"
_CHAT_SUGGESTION = "例如：激进一些、现金多一点、规划 3 年、不要重仓股票"
_CN_DIGITS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chat_prefs" not in st.session_state:
        st.session_state["chat_prefs"] = {}


def _extract_horizon(text: str) -> int | None:
    digit_match = re.search(r"(\d{1,2})\s*(?:年|yrs?|years?)", text, re.IGNORECASE)
    if digit_match:
        return int(digit_match.group(1))
    for cn, val in _CN_DIGITS.items():
        if f"{cn}年" in text:
            return val
    if "长期" in text or "long" in text.lower():
        return 8
    if "短期" in text or "short" in text.lower():
        return 2
    return None


def _parse_chat_preferences(text: str) -> Dict[str, Any]:
    prefs: Dict[str, any] = {}
    normalized = text.strip()
    if not normalized:
        return prefs
    lowered = normalized.lower()

    if any(keyword in normalized for keyword in ["激进", "进取", "高收益", "收益最大"]):
        prefs["risk_hint"] = "aggressive"
    elif any(keyword in normalized for keyword in ["稳健", "保守", "不要亏", "稳一点"]):
        prefs["risk_hint"] = "conservative"
    elif any(keyword in lowered for keyword in ["aggressive", "growth"]):
        prefs["risk_hint"] = "aggressive"
    elif any(keyword in lowered for keyword in ["conservative", "safe", "cautious"]):
        prefs["risk_hint"] = "conservative"

    horizon = _extract_horizon(normalized)
    if horizon is not None:
        prefs["horizon_years"] = horizon

    cash_focus = any(keyword in normalized for keyword in ["现金", "存款", "流动性", "cash", "liquidity"])
    equity_focus = any(keyword in normalized for keyword in ["股票", "权益", "equity", "stock"])
    if cash_focus and not equity_focus:
        prefs["equity_cap"] = 0.4
    elif equity_focus and not cash_focus:
        prefs["equity_cap"] = 0.75

    return prefs


def _describe_prefs(prefs: Mapping[str, Any]) -> List[str]:
    tags: List[str] = []
    hint = prefs.get("risk_hint")
    if hint == "aggressive":
        tags.append("偏激进")
    elif hint == "conservative":
        tags.append("偏保守")
    horizon = prefs.get("horizon_years")
    if isinstance(horizon, (int, float)):
        tags.append(f"期限约 {int(horizon)} 年")
    equity_cap = prefs.get("equity_cap")
    if isinstance(equity_cap, (int, float)):
        tags.append(f"股票上限 {int(equity_cap * 100)}%")
    return tags


def _ingest_chat_message(text: str):
    msg = text.strip()
    if not msg:
        return
    st.session_state["chat_history"].append({"role": "user", "text": msg})
    parsed = _parse_chat_preferences(msg)
    if parsed:
        st.session_state["chat_prefs"].update(parsed)
        tags = _describe_prefs(parsed)
        reply = "已识别偏好：" + ("、".join(tags) if tags else str(parsed))
    else:
        reply = "暂未识别出结构化偏好，可以换种说法试试。"
    st.session_state["chat_history"].append({"role": "assistant", "text": reply})


def _render_preference_chat(container):
    container.subheader("🗣️ 自然语言偏好")
    container.caption(f"提示：{_CHAT_SUGGESTION}")
    history_box = container.container()
    if not st.session_state["chat_history"]:
        history_box.info("还没有对话内容，试着描述你的计划或担忧。")
    else:
        for message in st.session_state["chat_history"]:
            prefix = "👤" if message["role"] == "user" else "🤖"
            history_box.markdown(f"{prefix} {message['text']}")

    with container.form("chat_form", clear_on_submit=True):
        user_text = st.text_input("告诉我们你的计划、期限或风险偏好", key="chat_form_input")
        submitted = st.form_submit_button("记录偏好")
    if submitted and user_text:
        _ingest_chat_message(user_text)
        st.rerun()

    pref_tags = _describe_prefs(st.session_state["chat_prefs"])
    if pref_tags:
        container.success("已记录偏好：" + "、".join(pref_tags))
    else:
        container.info("尚未识别到偏好，可在上方对话框进一步描述。")


def _predict_q_values(policy, state_vec: np.ndarray) -> np.ndarray:
    action_count = len(ALL_CARDS)
    state_batch = np.repeat(state_vec[None, :], action_count, axis=0)
    action_batch = np.arange(action_count, dtype=np.int64)
    return policy.predict_value(state_batch, action_batch)


@st.cache_resource(show_spinner=False)
def _load_snapshot():
    env = MarketEnv()
    latest_snapshot: MarketSnapshot = env.mkt_sshots[-1]
    return latest_snapshot


@st.cache_resource(show_spinner=True)
def _load_policy(model_key: str):
    config = MODEL_REGISTRY[model_key]
    policy = load_policy_artifact(str(config["path"]), require_gpu=False)
    return policy


def _collect_user_inputs(container) -> Dict:
    container.subheader("客户画像输入")

    age = container.slider("年龄", min_value=20, max_value=80, value=42)
    balance = int(container.number_input("可投资资产 (¥)", min_value=10000, max_value=5_000_000, value=500_000, step=50_000))
    job = container.selectbox(
        "职业",
        ["management", "technician", "admin.", "services", "retired", "student", "blue-collar"],
        index=0,
    )
    marital = container.selectbox("婚姻状况", ["single", "married", "divorced"], index=1)
    education = container.selectbox("教育水平", ["primary", "secondary", "tertiary", "unknown"], index=2)
    housing = container.radio("住房贷款", ["yes", "no"], index=1, horizontal=True)
    loan = container.radio("消费贷款", ["yes", "no"], index=1, horizontal=True)
    default = container.radio("历史违约", ["no", "yes"], index=0, horizontal=True)

    alloc_templates = {
        "稳健型 (40/40/20)": (0.4, 0.4, 0.2),
        "保守型 (20/30/50)": (0.2, 0.3, 0.5),
        "进取型 (60/30/10)": (0.6, 0.3, 0.1),
    }
    alloc_label = container.selectbox("当前组合", list(alloc_templates.keys()), index=0)
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


def render_recommendations(
    policy,
    snapshot,
    profile: UserProfile,
    current_alloc: np.ndarray,
    extra_prefs: Mapping[str, Any] | None = None,
):
    state_vec = build_state_vec(snapshot, profile, current_alloc)
    q_values = _predict_q_values(policy, state_vec)

    allowed_cards = allowed_cards_for_user(profile.risk_bucket)
    allowed_ids = [card.act_id for card in allowed_cards]

    merged_prefs = infer_prefs_from_profile(profile)
    if extra_prefs:
        merged_prefs.update(extra_prefs)

    priors = build_personal_prior(
        allowed_ids,
        prefs=merged_prefs,
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

    pref_tags = _describe_prefs(merged_prefs)
    if pref_tags:
        st.caption("个性化信号：" + "、".join(pref_tags))

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
        "离线策略 + 合规安全壳。输入客户画像即可查看推荐卡片、Q 值与审计摘要。"
    )

    model_options = [key for key in MODEL_ORDER if key in MODEL_REGISTRY] or list(MODEL_REGISTRY.keys())
    default_index = max(0, model_options.index(DEFAULT_MODEL_KEY)) if DEFAULT_MODEL_KEY in model_options else 0
    selected_model = st.sidebar.selectbox(
        "选择策略模型",
        model_options,
        index=default_index,
        format_func=lambda key: MODEL_REGISTRY[key]["label"],
    )
    try:
        policy = _load_policy(selected_model)
    except FileNotFoundError:
        st.error(f"未找到模型文件：{MODEL_REGISTRY[selected_model]['path']}")
        st.stop()
    except Exception as exc:
        st.error(f"模型加载失败：{exc}")
        st.stop()

    snapshot = _load_snapshot()

    _init_session_state()

    tab_reco, tab_analysis = st.tabs(["实时推荐", "千人千面分析"])
    with tab_reco:
        col_left, col_right = st.columns([0.42, 0.58], gap="large")
        inputs = _collect_user_inputs(col_left)
        col_left.info(
            f"当前策略：{MODEL_REGISTRY[selected_model]['label']} · 风险等级 {inputs['profile'].risk_bucket}",
            icon="🛡️",
        )
        col_left.metric("风险等级 (0=保守,2=进取)", inputs["profile"].risk_bucket)
        col_left.metric(
            "当前配置",
            _format_percentage_vector(inputs["current_alloc"]),
            help="用于拼接状态向量，也可作为组合调仓参考。",
        )
        _render_preference_chat(col_left)

        with col_right:
            render_recommendations(
                policy,
                snapshot,
                profile=inputs["profile"],
                current_alloc=inputs["current_alloc"],
                extra_prefs=st.session_state["chat_prefs"],
            )

    with tab_analysis:
        render_analysis_tab()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "✅ 当前界面仅用于演示，不会触发真实交易。\n\n"
        "☑️ 可在顶部选择不同策略模型做对比。"
    )


if __name__ == "__main__":
    main()
