#!/usr/bin/env bash

# End-to-end pipeline：统一 reward scale、重跑训练并保存评估结果。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=./src

DATA="${DATA:-./data/offline_dataset_reward_personal.h5}"
META="${META:-${DATA%.*}_behavior.npz}"
REPORT_DIR="${REPORT_DIR:-./reports}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./artifacts/pipeline_eval}"
mkdir -p "$REPORT_DIR" "$ARTIFACT_DIR"

REWARD_SCALE="${REWARD_SCALE:-1000}"
CQL_STEPS="${CQL_STEPS:-800000}"
EPSILON="${EPSILON:-0.2}"
NUM_USERS="${NUM_USERS:-45000}"
EPISODE_STEPS="${EPISODE_STEPS:-252}"
FQE_STEPS="${FQE_STEPS:-200000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20000}"

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
  local log_path="$ARTIFACT_DIR/${algo}.log"
  local summary_path="$ARTIFACT_DIR/${algo}.summary.json"

  step "3) 评估 ${algo}"
  python -m hybrid_advisor_offline.offline.eval.eval_policy \
    --dataset "$DATA" \
    --model "$model" \
    --behavior-meta "$META" \
    --fqe-steps "$FQE_STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    | tee "$log_path"

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
  cleanup
}

step "1) 生成离线数据集"
python -m hybrid_advisor_offline.offline.trainrl.gen_datasets \
  --dataset-path "$DATA" \
  --num-users "$NUM_USERS" \
  --episode-steps "$EPISODE_STEPS" \
  --epsilon "$EPSILON"
cleanup

step "2) 训练 BC (reward_scale=${REWARD_SCALE})"
python -m hybrid_advisor_offline.offline.trainrl.train_bc \
  --dataset "$DATA" \
  --model-output ./models/bc_reward_personal.pt \
  --reward-scale "$REWARD_SCALE" \
  --require-gpu
cleanup

step "2) 训练 BCQ (reward_scale=${REWARD_SCALE})"
python -m hybrid_advisor_offline.offline.trainrl.train_bcq \
  --dataset "$DATA" \
  --model-output ./models/bcq_reward_personal.pt \
  --reward-scale "$REWARD_SCALE" \
  --require-gpu
cleanup

step "2) 训练 CQL (reward_scale=${REWARD_SCALE})"
python -m hybrid_advisor_offline.offline.trainrl.train_cql \
  --dataset "$DATA" \
  --model-output ./models/cql_reward_personal.pt \
  --steps "$CQL_STEPS" \
  --reward-scale "$REWARD_SCALE" \
  --require-gpu
cleanup

run_eval "bc" "./models/bc_reward_personal.pt"
run_eval "bcq" "./models/bcq_reward_personal.pt"
run_eval "cql" "./models/cql_reward_personal.pt"

step "4) 生成分群对比图表"
python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/bc_reward_personal.pt \
  --algo-name bc_reward_personal \
  --output-csv "$REPORT_DIR/segment_metrics_bc.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_bc.png"
cleanup

python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/bcq_reward_personal.pt \
  --algo-name bcq_reward_personal \
  --output-csv "$REPORT_DIR/segment_metrics_bcq.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_bcq.png"
cleanup

python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/cql_reward_personal.pt \
  --algo-name cql_reward_personal \
  --output-csv "$REPORT_DIR/segment_metrics_cql.csv" \
  --output-fig "$REPORT_DIR/segment_metrics_cql.png"
cleanup

echo -e "\n✅ 全流程完成。评估日志与 JSON 摘要位于 ${ARTIFACT_DIR}"
