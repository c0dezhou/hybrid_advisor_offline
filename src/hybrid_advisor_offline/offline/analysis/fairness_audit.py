import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

SMALL_SAMPLE_THR = 100


def band_age(age: Any):
    try:
        age = int(age)
    except (TypeError, ValueError):
        return None
    if age <= 35:
        return "<=35"
    if age <= 55:
        return "36-55"
    return "56+"


def summarize(df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if group_col not in df.columns:
        return out
    for key, group in df.dropna(subset=[group_col]).groupby(group_col):
        n = int(len(group))
        total_return = group["total_return"]
        std_return = float(total_return.std(ddof=1)) if n > 1 else 0.0
        sem_return = float(std_return / np.sqrt(n)) if n > 0 else 0.0
        summary: dict[str, Any] = {
            "n_episodes": n,
            "avg_total_return": float(total_return.mean()),
            "std_total_return": std_return,
            "sem_total_return": sem_return,
            "avg_max_drawdown": float(group["max_drawdown"].mean()),
            "avg_sharpe": float(group["sharpe"].mean()),
            "avg_cvar5": float(group["cvar5"].mean()),
            "small_sample": n < SMALL_SAMPLE_THR,
        }
        if "accept_prob" in group.columns:
            summary["avg_accept_prob"] = float(group["accept_prob"].mean())
        out[str(key)] = summary
    return out


def pairwise_gap(metric_dict: dict[str, Any], metric: str, thr: float):
    keys = sorted(metric_dict.keys())
    flags = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = metric_dict[keys[i]].get(metric)
            b = metric_dict[keys[j]].get(metric)
            if a is None or b is None:
                continue
            gap = abs(a - b)
            flags.append(
                {"pair": f"{keys[i]} vs {keys[j]}", "gap": gap, "ok": gap <= thr}
            )
    overall_ok = all(flag["ok"] for flag in flags) if flags else True
    return flags, overall_ok


def _normalize_profiles(raw_profiles: np.ndarray | None) -> list[dict[str, Any]]:
    if raw_profiles is None:
        return []
    normalized = []
    for entry in raw_profiles:
        if isinstance(entry, dict):
            payload = entry
        else:
            try:
                payload = entry.item()
            except Exception:
                try:
                    payload = dict(entry)
                except Exception:
                    continue
        if not isinstance(payload, dict):
            continue
        normalized.append(payload)
    return normalized


def main() -> None:
    ap = argparse.ArgumentParser(
        description="按人群/风险桶做回测差异报告。",
    )
    ap.add_argument("--backtest-csv", required=True)
    ap.add_argument("--profiles-npz", required=False)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--gap-thr-maxdd", type=float, default=0.03)
    args = ap.parse_args()

    df = pd.read_csv(args.backtest_csv)
    if args.profiles_npz:
        try:
            with np.load(args.profiles_npz, allow_pickle=True) as meta:
                profiles = _normalize_profiles(meta.get("user_profiles"))
        except Exception as exc:
            print(f"[fairness_audit] 无法读取 profiles：{exc}")
            profiles = []
        if profiles:
            pf = (
                pd.DataFrame(profiles)
                .reset_index()
                .rename(columns={"index": "episode_id"})
            )
            df = df.merge(pf, on="episode_id", how="left")

    if "age" in df.columns:
        df["age_band"] = df["age"].map(band_age)
    else:
        df["age_band"] = None

    res = {
        "by_risk_bucket": summarize(df, "risk_bucket"),
        "by_age_band": summarize(df, "age_band"),
    }

    flags_rb, ok_rb = pairwise_gap(
        res["by_risk_bucket"], "avg_max_drawdown", args.gap_thr_maxdd
    )
    flags_ab, ok_ab = pairwise_gap(
        res["by_age_band"], "avg_max_drawdown", args.gap_thr_maxdd
    )

    res["checks"] = {
        "maxdd_gap_risk_bucket": {
            "pairs": flags_rb,
            "ok": ok_rb,
            "thr": args.gap_thr_maxdd,
        },
        "maxdd_gap_age_band": {
            "pairs": flags_ab,
            "ok": ok_ab,
            "thr": args.gap_thr_maxdd,
        },
        "todo_stat_test": True,
    }

    report_dir = os.path.dirname(args.report_json)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as fp:
        json.dump(res, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
