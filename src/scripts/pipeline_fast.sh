#!/usr/bin/env bash

# Fast-mode pipeline: generate a lightweight dataset, train BC/BCQ/CQL with reduced steps,
# and run eval_policy in --fast-dev mode for quick iteration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=./src

# Fast switches for generators / trainers
export USE_GPU=1
export EXPERIMENT_MODE="${EXPERIMENT_MODE:-fast}"
export FAST_MODE_USER_CAP="${FAST_MODE_USER_CAP:-2000}"
export FAST_MODE_STEP_CAP="${FAST_MODE_STEP_CAP:-64}"

DATA="${DATA:-./data/offline_dataset_fast.h5}"
META="${META:-${DATA%.*}_behavior.npz}"
REPORT_DIR="${REPORT_DIR:-./reports/fast}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./artifacts/pipeline_fast}"
mkdir -p "$REPORT_DIR" "$ARTIFACT_DIR"

NUM_USERS="${NUM_USERS:-2000}"
EPISODE_STEPS="${EPISODE_STEPS:-64}"
EPSILON="${EPSILON:-0.2}"
REWARD_SCALE="${REWARD_SCALE:-500}"
BC_STEPS="${BC_STEPS:-100000}"
BCQ_STEPS="${BCQ_STEPS:-150000}"
CQL_STEPS="${CQL_STEPS:-200000}"
FQE_STEPS="${FQE_STEPS:-50000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.05}"

USE_GPU="${USE_GPU:-0}"
GPU_FLAG=()
if [[ "$USE_GPU" == "1" ]]; then
  GPU_FLAG+=(--require-gpu)
fi

step() {
  echo -e "\n========== $* =========="
}

run_eval() {
  local algo="$1"
  local model="$2"
  shift 2
  local extra_args=("$@")
  local log_path="$ARTIFACT_DIR/${algo}.log"
  local summary_path="$ARTIFACT_DIR/${algo}.summary.json"

  step "评估 ${algo} (fast-dev)"
  local cmd=(
    python -m hybrid_advisor_offline.offline.eval.eval_policy
    --dataset "$DATA"
    --model "$model"
    --behavior-meta "$META"
    --fast-dev
    --fqe-steps "$FQE_STEPS"
    --eval-interval "$EVAL_INTERVAL"
    --validation-ratio "$VALIDATION_RATIO"
    "${GPU_FLAG[@]}"
  )
  if [[ ${#extra_args[@]} -gt 0 ]]; then
    cmd+=("${extra_args[@]}")
  fi
  "${cmd[@]}" | tee "$log_path"

  python - "$log_path" "$summary_path" <<'PY'
import json, pathlib, sys
log_path, out_path = sys.argv[1:]
text = pathlib.Path(log_path).read_text(encoding="utf-8")
marker = "=== 评估结果 ==="
idx = text.rfind(marker)
if idx == -1:
    raise SystemExit("无法在日志中找到评估结果段落")
snippet = text[idx + len(marker):].strip()
payload = json.loads(snippet)
pathlib.Path(out_path).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY

  echo "评估结果 JSON 已写入 ${summary_path}"
}

step "1) 生成 fast 模式离线数据集"
python -m hybrid_advisor_offline.offline.trainrl.gen_datasets \
  --dataset-path "$DATA" \
  --num-users "$NUM_USERS" \
  --episode-steps "$EPISODE_STEPS" \
  --epsilon "$EPSILON"

step "2) 训练 BC (fast-dev)"
python -m hybrid_advisor_offline.offline.trainrl.train_bc \
  --dataset "$DATA" \
  --model-output ./models/bc_fast.pt \
  --steps "$BC_STEPS" \
  --reward-scale "$REWARD_SCALE" \
  --fast-dev \
  "${GPU_FLAG[@]}"

step "2) 训练 BCQ (fast-dev)"
python -m hybrid_advisor_offline.offline.trainrl.train_bcq \
  --dataset "$DATA" \
  --model-output ./models/bcq_fast.pt \
  --steps "$BCQ_STEPS" \
  --reward-scale "$REWARD_SCALE" \
  --fast-dev \
  "${GPU_FLAG[@]}"

step "2) 训练 CQL (fast-dev)"
python -m hybrid_advisor_offline.offline.trainrl.train_cql \
  --dataset "$DATA" \
  --model-output ./models/cql_fast.pt \
  --steps "$CQL_STEPS" \
  --reward-scale "$REWARD_SCALE" \
  --fast-dev \
  "${GPU_FLAG[@]}"

run_eval "bc_fast" "./models/bc_fast.pt"
run_eval "bcq_fast" "./models/bcq_fast.pt" --backtest
BT_FAST_CSV="./models/bcq_fast.pt.backtest_val.csv"
FAIRNESS_FAST_REPORT="$REPORT_DIR/fairness_bcq_fast.json"
if [[ -f "$BT_FAST_CSV" ]]; then
  python -m hybrid_advisor_offline.offline.analysis.fairness_audit \
    --backtest-csv "$BT_FAST_CSV" \
    --profiles-npz "$META" \
    --report-json "$FAIRNESS_FAST_REPORT"
  echo "[pipeline-fast] 公平性审计报告已写入 $FAIRNESS_FAST_REPORT"
else
  echo "[pipeline-fast] 跳过公平性审计，缺少 $BT_FAST_CSV"
fi
run_eval "cql_fast" "./models/cql_fast.pt"

step "3) 生成 BCQ fast 分群图（可选）"
python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/bcq_fast.pt \
  --algo-name bcq_fast \
  --output-csv "$REPORT_DIR/segment_metrics_bcq_fast.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_bcq_fast.png"

echo -e "\n✅ fast 模式 pipeline 完成。日志位于 ${ARTIFACT_DIR}"
