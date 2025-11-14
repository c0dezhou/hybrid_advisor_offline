#!/usr/bin/env bash

# End-to-end pipeline：统一 reward scale、重跑训练并保存评估结果。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=./src
export USE_CARD_FACTORY=1

TRAIN_DATA="${TRAIN_DATA:-./data/offline_dataset_train.h5}"
VAL_DATA="${VAL_DATA:-./data/offline_dataset_val.h5}"
TRAIN_META="${TRAIN_META:-${TRAIN_DATA%.*}_behavior.npz}"
VAL_META="${VAL_META:-${VAL_DATA%.*}_behavior.npz}"
REPORT_DIR="${REPORT_DIR:-./reports}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./artifacts/pipeline_eval}"
export REPORT_DIR ARTIFACT_DIR
mkdir -p "$REPORT_DIR" "$ARTIFACT_DIR"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export BASELINE_EVAL="${BASELINE_EVAL:-1}"
python - <<'PY'
import importlib, json, os, platform, subprocess, sys

def ver(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as exc:
        return f"missing:{exc}"

info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": ver("torch"),
    "d3rlpy": ver("d3rlpy"),
    "numpy": ver("numpy"),
    "pandas": ver("pandas"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
}
try:
    info["git_commit"] = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )
except Exception:
    info["git_commit"] = "unknown"

out = os.path.join(os.environ["ARTIFACT_DIR"], "_env.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
print(f"[env] 已写入 {out}")
PY
SPLIT_META_PATH="${ARTIFACT_DIR}/_split_meta.json"

REWARD_SCALE="${REWARD_SCALE:-1000}"
CQL_STEPS="${CQL_STEPS:-500000}"
BC_STEPS="${BC_STEPS:-250000}"
BCQ_STEPS="${BCQ_STEPS:-500000}"
EPSILON="${EPSILON:-0.2}"
NUM_USERS="${NUM_USERS:-45000}"
EPISODE_STEPS="${EPISODE_STEPS:-252}"
VAL_NUM_USERS="${VAL_NUM_USERS:-$NUM_USERS}"
VAL_EPISODE_STEPS="${VAL_EPISODE_STEPS:-$EPISODE_STEPS}"
VAL_EPSILON="${VAL_EPSILON:-$EPSILON}"
FQE_STEPS="${FQE_STEPS:-200000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.1}"
FAST_RUN="${FAST_RUN:-0}"
FAST_NUM_USERS="${FAST_NUM_USERS:-2000}"
FAST_EPISODE_STEPS="${FAST_EPISODE_STEPS:-64}"
FAST_VAL_NUM_USERS="${FAST_VAL_NUM_USERS:-$FAST_NUM_USERS}"
FAST_VAL_EPISODE_STEPS="${FAST_VAL_EPISODE_STEPS:-$FAST_EPISODE_STEPS}"
FAST_FQE_STEPS="${FAST_FQE_STEPS:-50000}"
FAST_EVAL_INTERVAL="${FAST_EVAL_INTERVAL:-5000}"
FAST_VALIDATION_RATIO="${FAST_VALIDATION_RATIO:-0.1}"
FAST_CQL_STEPS="${FAST_CQL_STEPS:-50000}"
FAST_BC_STEPS="${FAST_BC_STEPS:-10000}"
FAST_BCQ_STEPS="${FAST_BCQ_STEPS:-15000}"

if [[ "$FAST_RUN" == "1" ]]; then
  NUM_USERS="$FAST_NUM_USERS"
  EPISODE_STEPS="$FAST_EPISODE_STEPS"
  VAL_NUM_USERS="$FAST_VAL_NUM_USERS"
  VAL_EPISODE_STEPS="$FAST_VAL_EPISODE_STEPS"
  FQE_STEPS="$FAST_FQE_STEPS"
  EVAL_INTERVAL="$FAST_EVAL_INTERVAL"
  VALIDATION_RATIO="$FAST_VALIDATION_RATIO"
  CQL_STEPS="$FAST_CQL_STEPS"
  BC_STEPS="$FAST_BC_STEPS"
  BCQ_STEPS="$FAST_BCQ_STEPS"
  EXTRA_EVAL_ARGS=(--fast-dev)
  echo "[pipeline_with_sleep] FAST_RUN=1 -> 精简规模: users=${NUM_USERS}, steps=${EPISODE_STEPS}, FQE_steps=${FQE_STEPS}"
else
  EXTRA_EVAL_ARGS=()
fi
EMBARGO_DAYS="${EMBARGO_DAYS:-20}"
export EPISODE_EMBARGO_DAYS="${EMBARGO_DAYS}"
MAX_EPISODE_STEPS=$(( EPISODE_STEPS > VAL_EPISODE_STEPS ? EPISODE_STEPS : VAL_EPISODE_STEPS ))
DEFAULT_BLOCK_SIZE=$(( MAX_EPISODE_STEPS + EMBARGO_DAYS * 3 ))
START_BLOCK_SIZE="${START_BLOCK_SIZE:-$DEFAULT_BLOCK_SIZE}"
TRAIN_START_FILTER="${TRAIN_START_FILTER:-even}"
VAL_START_FILTER="${VAL_START_FILTER:-odd}"
TRAIN_START_SEED="${TRAIN_START_SEED:-100}"
VAL_START_SEED="${VAL_START_SEED:-200}"

NUM_DAYS="$(
  python - <<'PY'
import pandas as pd, pathlib, sys

path = pathlib.Path("./data/mkt_data.csv")
if not path.exists():
    sys.stdout.write("0")
    sys.exit(0)
df = pd.read_csv(path, usecols=[0])
sys.stdout.write(str(len(df)))
PY
)"
if [[ -z "$NUM_DAYS" || "$NUM_DAYS" -le 0 ]]; then
  echo "[pipeline_with_sleep] 无法读取 data/mkt_data.csv，默认 NUM_DAYS=5207"
  NUM_DAYS=5207
fi
MAX_START_IDX=$(( NUM_DAYS - MAX_EPISODE_STEPS ))
if [[ "$MAX_START_IDX" -lt 0 ]]; then
  MAX_START_IDX=0
fi
MAX_START_EXCLUSIVE=$(( MAX_START_IDX + 1 ))
GAP=$(( EPISODE_STEPS + EMBARGO_DAYS * 2 ))
if [[ "$MAX_START_IDX" -le "$GAP" ]]; then
  TRAIN_RANGE_END=1
else
  TRAIN_RANGE_END=$(( (MAX_START_IDX - GAP) / 2 ))
fi
if [[ "$TRAIN_RANGE_END" -lt 1 ]]; then
  TRAIN_RANGE_END=1
fi
VAL_RANGE_START=$(( TRAIN_RANGE_END + GAP ))
if [[ "$VAL_RANGE_START" -ge "$MAX_START_EXCLUSIVE" ]]; then
  VAL_RANGE_START=$(( MAX_START_EXCLUSIVE - 1 ))
  if [[ "$VAL_RANGE_START" -le "$TRAIN_RANGE_END" ]]; then
    VAL_RANGE_START=$(( TRAIN_RANGE_END + 1 ))
  fi
fi
TRAIN_START_RANGE="0:${TRAIN_RANGE_END}"
VAL_START_RANGE="${VAL_RANGE_START}:${MAX_START_EXCLUSIVE}"
echo "[pipeline_with_sleep] EPISODE_START_RANGE train=${TRAIN_START_RANGE}, val=${VAL_START_RANGE}, GAP=${GAP}"

step() {
  echo -e "\n========== $* =========="
}

cleanup() {
  python - <<'PY'
import gc; gc.collect()
PY
  sleep 5
}

run_eval() {
  local algo="$1"
  local model="$2"
  local dataset="$3"
  local meta="$4"
  shift 4
  local extra_args=("$@")
  local log_path="$ARTIFACT_DIR/${algo}.log"
  local summary_path="$ARTIFACT_DIR/${algo}.summary.json"

  step "3) 评估 ${algo}"
  echo "[sanity] 期望评估读取 reward_scale=${REWARD_SCALE}（来自模型配置）"
  local cmd=(python -m hybrid_advisor_offline.offline.eval.eval_policy
    --dataset "$dataset"
    --behavior-meta "$meta"
    --fqe-steps "$FQE_STEPS"
    --eval-interval "$EVAL_INTERVAL"
    --validation-ratio "$VALIDATION_RATIO"
    --summary-output "$summary_path"
  )
  if [[ -n "$model" ]]; then
    cmd+=(--model "$model")
  else
    cmd+=(--model "")
  fi
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  if [[ ${#EXTRA_EVAL_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_EVAL_ARGS[@]}")
  fi
  "${cmd[@]}" | tee "$log_path"

  echo "评估结果 JSON 已写入 ${summary_path}"
  cleanup
}

# step "1a) 生成训练集 (start_filter=${TRAIN_START_FILTER}, seed=${TRAIN_START_SEED})"
# EPISODE_START_RANGE="${TRAIN_START_RANGE}" \
# EPISODE_START_SEED="$TRAIN_START_SEED" \
# python -m hybrid_advisor_offline.offline.trainrl.gen_datasets \
#   --dataset-path "$TRAIN_DATA" \
#   --num-users "$NUM_USERS" \
#   --episode-steps "$EPISODE_STEPS" \
#   --epsilon "$EPSILON" \
#   --split-name train \
#   --start-filter "$TRAIN_START_FILTER" \
#   --start-block-size "$START_BLOCK_SIZE"
# cleanup

# step "1b) 生成验证集 (start_filter=${VAL_START_FILTER}, seed=${VAL_START_SEED})"
# EPISODE_START_RANGE="${VAL_START_RANGE}" \
# EPISODE_START_SEED="$VAL_START_SEED" \
# python -m hybrid_advisor_offline.offline.trainrl.gen_datasets \
#   --dataset-path "$VAL_DATA" \
#   --num-users "$VAL_NUM_USERS" \
#   --episode-steps "$VAL_EPISODE_STEPS" \
#   --epsilon "$VAL_EPSILON" \
#   --split-name val \
#   --start-filter "$VAL_START_FILTER" \
#   --start-block-size "$START_BLOCK_SIZE"
# cleanup

# step "1c) 检查 train/val 时间窗是否重叠 (embargo=${EMBARGO_DAYS})"
# if [[ "$FAST_RUN" == "1" ]]; then
#   echo "[split_check] FAST_RUN=1，跳过 train/val 重叠校验。"
# else
#   python - "$TRAIN_META" "$VAL_META" "$EPISODE_STEPS" "$VAL_EPISODE_STEPS" "$EMBARGO_DAYS" "$START_BLOCK_SIZE" "$SPLIT_META_PATH" <<'PY'
# import json
# import os
# import sys
# import numpy as np

# train_meta, val_meta, default_train_steps, default_val_steps, embargo, block_size, report_path = sys.argv[1:]
# default_train_steps = int(default_train_steps)
# default_val_steps = int(default_val_steps)
# embargo = int(embargo)
# block_size = int(block_size)

# def load_meta(path, default_steps, label):
#     if not os.path.exists(path):
#         raise SystemExit(f"[split_check] 找不到 {label} 行为元数据：{path}")
#     data = np.load(path, allow_pickle=True)
#     starts = data.get("episode_start_idx")
#     if starts is None:
#         raise SystemExit(f"[split_check] {path} 缺少 episode_start_idx，无法校验时间窗。")
#     steps_arr = data.get("episode_steps")
#     steps = int(steps_arr[0]) if steps_arr is not None and len(steps_arr) else default_steps
#     return {
#         "starts": np.sort(starts.astype(np.int64)),
#         "steps": steps,
#     }

# def to_ranges(starts, steps, embargo_days):
#     width = steps + embargo_days
#     return [(int(s), int(s) + width) for s in starts]

# def scan_gap(r1, r2):
#     i = j = 0
#     min_gap = None
#     while i < len(r1) and j < len(r2):
#         a1, b1 = r1[i]
#         a2, b2 = r2[j]
#         if b1 <= a2:
#             gap = a2 - b1
#             min_gap = gap if min_gap is None else min(min_gap, gap)
#             i += 1
#         elif b2 <= a1:
#             gap = a1 - b2
#             min_gap = gap if min_gap is None else min(min_gap, gap)
#             j += 1
#         else:
#             return True, None
#     return False, (min_gap if min_gap is not None else 0)

# train = load_meta(train_meta, default_train_steps, "train")
# val = load_meta(val_meta, default_val_steps, "val")

# train_ranges = to_ranges(train["starts"], train["steps"], embargo)
# val_ranges = to_ranges(val["starts"], val["steps"], embargo)

# overlap, min_gap = scan_gap(train_ranges, val_ranges)
# start_overlap = len(np.intersect1d(train["starts"], val["starts"]))

# report = {
#     "train_meta": train_meta,
#     "val_meta": val_meta,
#     "episode_steps_train": train["steps"],
#     "episode_steps_val": val["steps"],
#     "embargo_days": embargo,
#     "start_block_size": block_size,
#     "n_train_episodes": int(len(train["starts"])),
#     "n_val_episodes": int(len(val["starts"])),
#     "start_idx_overlap": int(start_overlap),
#     "start_idx_overlap_ok": bool(start_overlap == 0),
#     "time_overlap": bool(overlap),
#     "min_buffer_days": int(min_gap if min_gap is not None else 0),
#     "min_buffer_ok": bool((min_gap if min_gap is not None else 0) >= embargo),
# }

# if overlap or start_overlap > 0:
#     print(json.dumps(report, ensure_ascii=False, indent=2))
#     raise SystemExit(
#         "❌ Train/Val 时间区间或起点集合发生重叠，请增大 START_BLOCK_SIZE 或 EMBARGO_DAYS。"
#     )

# os.makedirs(os.path.dirname(report_path), exist_ok=True)
# with open(report_path, "w", encoding="utf-8") as fp:
#     json.dump(report, fp, ensure_ascii=False, indent=2)

# print(
#     f"[split_check] OK：train/val 无重叠，最小缓冲 {report['min_buffer_days']} 天（包含 embargo={embargo}），"
#     f"起点集合交集 {report['start_idx_overlap']}。"
# )
# PY
# fi
# cleanup

# step "1d) 补齐行为元数据字段"
# python -m scripts.fix_behavior_meta \
#   --behavior-meta "$TRAIN_META" \
#   --output "$TRAIN_META" \
#   --rule-eps "$EPSILON"
# python -m scripts.fix_behavior_meta \
#   --behavior-meta "$VAL_META" \
#   --output "$VAL_META" \
#   --rule-eps "$VAL_EPSILON"
# cleanup

step "2) 训练 BC (reward_scale=${REWARD_SCALE})"
python -m hybrid_advisor_offline.offline.trainrl.train_bc \
  --dataset "$TRAIN_DATA" \
  --model-output ./models/bc_reward_personal.pt \
  --reward-scale "$REWARD_SCALE" \
  --steps "$BC_STEPS" \
  --require-gpu
cleanup

step "2) 训练 BCQ (reward_scale=${REWARD_SCALE})"
python -m hybrid_advisor_offline.offline.trainrl.train_bcq \
  --dataset "$TRAIN_DATA" \
  --model-output ./models/bcq_reward_personal.pt \
  --reward-scale "$REWARD_SCALE" \
  --steps "$BCQ_STEPS" \
  --require-gpu
cleanup

step "2) 训练 CQL (reward_scale=${REWARD_SCALE})"
python -m hybrid_advisor_offline.offline.trainrl.train_cql \
  --dataset "$TRAIN_DATA" \
  --model-output ./models/cql_reward_personal.pt \
  --steps "$CQL_STEPS" \
  --reward-scale "$REWARD_SCALE" \
  --require-gpu
cleanup

run_eval "bc" "./models/bc_reward_personal.pt" "$VAL_DATA" "$VAL_META" --backtest
BC_BACKTEST="./models/bc_reward_personal.pt.backtest_val.csv"
if [[ -f "$BC_BACKTEST" ]]; then
  python -m hybrid_advisor_offline.offline.analysis.fairness_audit \
    --backtest-csv "$BC_BACKTEST" \
    --profiles-npz "$VAL_META" \
    --report-json "$REPORT_DIR/bc.fairness.json"
  echo "[pipeline] 公平性审计报告已写入 $REPORT_DIR/bc.fairness.json"
else
  echo "[pipeline] 跳过公平性审计，缺少 $BC_BACKTEST"
fi

run_eval "bcq" "./models/bcq_reward_personal.pt" "$VAL_DATA" "$VAL_META" --backtest
BT_CSV="./models/bcq_reward_personal.pt.backtest_val.csv"
FAIRNESS_REPORT="$REPORT_DIR/fairness_bcq.json"
if [[ -f "$BT_CSV" ]]; then
  python -m hybrid_advisor_offline.offline.analysis.fairness_audit \
    --backtest-csv "$BT_CSV" \
    --profiles-npz "$VAL_META" \
    --report-json "$FAIRNESS_REPORT"
  echo "[pipeline] 公平性审计报告已写入 $FAIRNESS_REPORT"
else
  echo "[pipeline] 跳过公平性审计，缺少 $BT_CSV"
fi

run_eval "cql" "./models/cql_reward_personal.pt" "$VAL_DATA" "$VAL_META" --backtest
CQL_BACKTEST="./models/cql_reward_personal.pt.backtest_val.csv"
if [[ -f "$CQL_BACKTEST" ]]; then
  python -m hybrid_advisor_offline.offline.analysis.fairness_audit \
    --backtest-csv "$CQL_BACKTEST" \
    --profiles-npz "$VAL_META" \
    --report-json "$REPORT_DIR/cql.fairness.json"
  echo "[pipeline] 公平性审计报告已写入 $REPORT_DIR/cql.fairness.json"
else
  echo "[pipeline] 跳过公平性审计，缺少 $CQL_BACKTEST"
fi

if [[ "$BASELINE_EVAL" == "1" ]]; then
  run_eval "rule_baseline" "" "$VAL_DATA" "$VAL_META"
  run_eval "random_baseline" "" "$VAL_DATA" "$VAL_META"
fi

step "4) 生成分群对比图表"
python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$VAL_DATA" \
  --behavior-meta "$VAL_META" \
  --model ./models/bc_reward_personal.pt \
  --algo-name bc_reward_personal \
  --output-csv "$REPORT_DIR/segment_metrics_bc.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_bc.png"
cleanup

python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$VAL_DATA" \
  --behavior-meta "$VAL_META" \
  --model ./models/bcq_reward_personal.pt \
  --algo-name bcq_reward_personal \
  --output-csv "$REPORT_DIR/segment_metrics_bcq.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_bcq.png"
cleanup

python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$VAL_DATA" \
  --behavior-meta "$VAL_META" \
  --model ./models/cql_reward_personal.pt \
  --algo-name cql_reward_personal \
  --output-csv "$REPORT_DIR/segment_metrics_cql.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_cql.png"
cleanup

step "5) 汇总评估摘要到 CSV"
python - <<'PY'
import glob, json, os, pandas as pd

art = os.environ["ARTIFACT_DIR"]
rep = os.environ["REPORT_DIR"]
rows = []
for path in sorted(glob.glob(os.path.join(art, "*.summary.json"))):
    algo = os.path.basename(path).replace(".summary.json", "")
    data = json.load(open(path, encoding="utf-8"))
    fqe = data.get("fqe", {})
    cpe = data.get("cpe", {})
    risk = (data.get("risk_metrics_behavior") or {}).get("overall", {})
    rows.append({
        "algo": algo,
        "fqe_val_est_return": fqe.get("val_est_episode_return"),
        "fqe_train_est_return": fqe.get("train_est_episode_return"),
        "cpe_episode_return_mean": cpe.get("episode_return_mean"),
        "ips": cpe.get("ips"),
        "snips": cpe.get("snips"),
        "random_ips": cpe.get("random_baseline_ips"),
        "random_snips": cpe.get("random_baseline_snips"),
        "risk_total_return": risk.get("avg_total_return"),
        "risk_max_drawdown": risk.get("avg_max_drawdown"),
    })
df = pd.DataFrame(rows)
out = os.path.join(rep, "summary_table.csv")
df.to_csv(out, index=False)
print("[report] 汇总表已写入", out)
PY
cleanup

echo -e "\n✅ 全流程完成。评估日志与 JSON 摘要位于 ${ARTIFACT_DIR}"
