cd /home/zxy/hybrid_advisor_offline && PYTHONPATH=./src \
bash -lc '
set -euo pipefail

step () { echo -e "\n========== $* =========="; }
cleanup () {
  python - <<'"'"'PY'"'"'
import gc; gc.collect()
PY
  sleep 5
}

DATA=./data/offline_dataset_reward_personal.h5
META=./data/offline_dataset_reward_personal_behavior.npz

# step "1) 生成离线数据集"
# python -m hybrid_advisor_offline.offline.trainrl.gen_datasets \
#   --dataset-path "$DATA" \
#   --num-users 45000 --episode-steps 252 --epsilon 0.2
# cleanup

# step "2) 训练 BC / BCQ / CQL"
# python -m hybrid_advisor_offline.offline.trainrl.train_bc \
#   --dataset "$DATA" \
#   --model-output ./models/bc_reward_personal.pt --require-gpu
# cleanup

# python -m hybrid_advisor_offline.offline.trainrl.train_bcq \
#   --dataset "$DATA" \
#   --model-output ./models/bcq_reward_personal.pt --require-gpu
# cleanup

# python -m hybrid_advisor_offline.offline.trainrl.train_cql \
#   --dataset "$DATA" \
#   --model-output ./models/cql_reward_personal.pt \
#   --steps 800000 --alpha 1.0 --learning-rate 3e-4 \
#   --reward-scale 1000 --require-gpu
# cleanup

# step "3) 逐个评估（FQE + CPE + max drawdown）"
# for algo in bc bcq cql; do
#   case "$algo" in
#     bc)  MODEL=./models/bc_reward_personal.pt ;;
#     bcq) MODEL=./models/bcq_reward_personal.pt ;;
#     cql) MODEL=./models/cql_reward_personal.pt ;;
#   esac
#   python -m hybrid_advisor_offline.offline.eval.eval_policy \
#     --dataset "$DATA" \
#     --model "$MODEL" \
#     --behavior-meta "$META" \
#     --fqe-steps 200000 --eval-interval 20000
#   cleanup
# done

step "4) 生成分群对比图表"
python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/bc_reward_personal.pt \
  --algo-name bc_reward_personal \
  --output-csv ./reports/segment_metrics_bc.csv \
  --output-fig ./reports/segment_metrics_bc.png
cleanup

python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/bcq_reward_personal.pt \
  --algo-name bcq_reward_personal \
  --output-csv ./reports/segment_metrics_bcq.csv \
  --output-fig ./reports/segment_metrics_bcq.png
cleanup

python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset "$DATA" \
  --behavior-meta "$META" \
  --model ./models/cql_reward_personal.pt \
  --algo-name cql_reward_personal \
  --output-csv ./reports/segment_metrics_cql.csv \
  --output-fig ./reports/segment_metrics_cql.png
cleanup

echo -e "\n✅ 全流程完成，报告见 ./reports/"
'
