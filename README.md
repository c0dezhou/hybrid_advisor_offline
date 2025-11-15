Hybrid Advisor Offline
======================

混合式合规投顾离线实验框架。项目内已经包含数据生成与 **BCQ 为主** 的离线策略训练与评估脚本（同时保留 BC / CQL 作为对比基线），并提供了可切换“客户版 / 审计版”的前端演示与影子接口，方便将离线策略接入业务或审计流程。

一键部署脚本：src/scripts/pipeline_with_sleep.sh
执行一键部署：
chmod +x pipeline_with_sleep.sh
./pipeline_with_sleep.sh

======================

依赖准备
--------

```bash
conda create -n advisor python=3.10
conda activate advisor
pip install -r requirements.txt
```

`requirements.txt` 已覆盖：

- 离线训练 / 评估：`numpy`, `pandas`, `torch`, `d3rlpy`, `gym` 等；
- 前端 / 影子接口：`streamlit`, `fastapi`, `uvicorn[standard]`；
- 可选 LLM 文案润色：`transformers`, `huggingface_hub`, `accelerate`, `bitsandbytes`, `llama-cpp-python`。

如需在 GPU 服务器上使用特定版本的 `torch` 或 `d3rlpy`，可在安装前先按官方指引单独安装对应版本，再执行 `pip install -r requirements.txt`。

运行 Streamlit Demo
-------------------

1. 先按照 `src/hybrid_advisor_offline/offline/trainrl` 下的脚本（或 `src/scripts/pipeline_with_sleep.sh`）准备数据，**至少训练出 `bcq_reward_personal.pt`（推荐）**；如需做对比，可额外训练 `bc_reward_personal.pt` / `cql_reward_personal.pt` 等模型。
2. 可选：导出推理专用模型文件，加速前端加载（见下一节，BCQ 已提供示例）。
3. 运行以下命令启动本地 UI（默认监听 8501 端口）：

```bash
在项目根目录：
streamlit run src/hybrid_advisor_offline/ux/demo_streamlit.py
```

左侧面板用于输入客户画像与目标组合、记录自然语言偏好；右侧支持**客户版**与**审计版**两种视图：

- 客户版：仅展示推荐的 Top-3 调仓卡片、目标配置与风险等级，以及面向客户的解释文案；
- 审计版：在客户版信息基础上，额外展示合规可用动作集合的 Q 值分布、每张推荐卡片的模型 Q 值与审计摘要哈希，并暴露原始画像/状态向量以便核查。

界面右上角的模式切换位于侧边栏中的“展示模式”单选框。

导出推理专用模型文件（可选，但推荐）
--------------------------------------

为了降低前端冷启动时间，Demo 会优先尝试加载“推理专用”策略文件（仅包含网络结构、参数与 scaler），避免在 UI 线程中重复读取大规模 ReplayBuffer 并重新拟合 scaler。

训练完成后，可通过以下脚本导出对应的推理文件（以 BCQ 为例）：

```bash
export PYTHONPATH=./src
python src/scripts/export_inference_policy.py \
  --model ./models/bcq_reward_personal.pt
```

默认会在同目录生成 `./models/bcq_reward_personal.pt.inference.d3`。  
前端会按以下顺序尝试加载策略：

1. 若存在 `*.pt.inference.d3`：通过 `load_policy_inference` 走轻量加载路径（推荐）；
2. 否则：回退到 `load_policy_artifact`，读取 ReplayBuffer 并重建训练期 scaler（较慢，但兼容旧环境）。

快速 Demo 模式
--------------

前端演示对加载速度比较敏感，可以通过“小数据 + 小模型”的 demo 方案来缩短启动时间（正式实验仍推荐使用完整数据 + BCQ）：

```bash
# 1. 从完整数据集中抽样 1 万条 transition
python -m hybrid_advisor_offline.offline.trainrl.make_demo_dataset \
  --source ./data/offline_dataset_reward_personal.h5 \
  --target ./data/offline_dataset_demo.h5 \
  --n-samples 10000

# 2. 用抽样数据训练一个轻量 Demo 模型（默认使用离散 CQL；如要与 BCQ 主线对齐，可改用 BCQ 训练脚本）
python -m hybrid_advisor_offline.offline.trainrl.train_discrete_demo \
  --dataset ./data/offline_dataset_demo.h5 \
  --model-output ./models/cql_demo.pt \
  --steps 100000 --reward-scale 300

# 3. 以 Demo 模式启动前端 / 影子 API（可搭配推理专用模型文件）
export DEMO_MODE=1
python -m streamlit run hybrid_advisor_offline/ux/demo_streamlit.py
# 或 uvicorn hybrid_advisor_offline.ux.shadow_api:app --host 0.0.0.0 --port 8000
```

取消 `DEMO_MODE`（或设置为 0）即可恢复到正式模型，新增的环境变量 `DEMO_DATASET_PATH`、`DEMO_MODEL_PATH` 也可以覆盖默认 demo 路径。

运行 FastAPI 影子接口
--------------------

```bash
export PYTHONPATH=./src  # 或 `pip install -e .`
uvicorn hybrid_advisor_offline.ux.shadow_api:app --host 0.0.0.0 --port 8000 --reload
```

接口说明：

* `GET /`：健康检查，返回模型加载状态与所用策略版本。
* `POST /recommend`：输入客户画像与当前配置，返回合规过滤后的推荐卡片、面向客户的解释文本，以及用于审计的 Q 值与哈希等内部字段（等价于 Demo 的“审计版”视图）。

影子接口默认在 CPU 上加载模型，若希望使用 GPU，可在调用 `load_policy_artifact` / `load_policy_inference` 时传入 `require_gpu=True`。

可选：启用 LLM 文案润色
------------------------

`demo_streamlit` 与 `shadow_api` 都会在 `USE_LLM_TRANSLATOR=1` 时尝试调用本地指令模型对客户提示语做润色：

```bash
# 安装依赖（按需，可使用不同模型）
pip install transformers huggingface_hub accelerate bitsandbytes llama-cpp-python

export USE_LLM_TRANSLATOR=1
# 默认使用仓库内的 GGUF（qwen2-7b-instruct-q5_k_m.gguf）
export LLM_TRANSLATOR_BACKEND=gguf
export LOCAL_TRANSLATOR_MODEL=./models/local_llm/qwen2-7b-instruct-q5_k_m.gguf
export GGUF_GPU_LAYERS=40

# 如果要改回 HF / bitsandbytes 版本，可自行下载模型后重设
# huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir ./models/local_llm/Qwen2-7B-Instruct
# export LLM_TRANSLATOR_BACKEND=hf
# export LOCAL_TRANSLATOR_MODEL=./models/local_llm/Qwen2-7B-Instruct
```

若未配置模型或推理失败，系统会自动回退到本地规则润色，保证输出始终符合合规措辞。 current translator 状态会在 Demo 页面中以提示条形式展示，同时 API 响应中也会附带 `translator_meta` 字段。

离线 BCQ 训练 + 评估（推荐主线）
-------------------------------

新增的 **BCQ 实验管线** 与现有的 DiscreteBC / DiscreteCQL 共用同一份 `offline_dataset.h5` 与 reward_scale，便于用 FQE / CPE 指标对齐比较。**在当前版本中，BCQ 是默认推荐的主力策略**。典型流程：

```bash
# 1) 生成或更新离线数据（存在即可跳过）
PYTHONPATH=./src \
python -m hybrid_advisor_offline.offline.trainrl.gen_datasets \
  --dataset-path ./data/offline_dataset_small.h5 \
  --num-users 5000 \
  --episode-steps 126 \
  --epsilon 0.2

# 2) 训练 BCQ（离散版）
USE_CARD_FACTORY=1 \
python -m hybrid_advisor_offline.offline.trainrl.train_bcq \
  --require-gpu \
  --dataset ./data/offline_dataset.h5 \
  --model-output ./models/bcq_discrete_model.pt \
  --steps 500000 \
  --learning-rate 1e-4 \
  --batch-size 256 \
  --reward-scale 1000

# 3) 复用 eval_policy 做 FQE + CPE 评估
USE_CARD_FACTORY=1 \
python -m hybrid_advisor_offline.offline.eval.eval_policy \
  --dataset ./data/offline_dataset.h5 \
  --model ./models/bcq_discrete_model.pt \
  --behavior-meta ./data/offline_dataset_behavior.npz \
  --fqe-steps 150000 \
  --eval-interval 20000
```

评估时重点关注 `fqe.val_est_episode_return` 与 `cpe.episode_return_mean`，目标是让 BCQ 的估计回报在同一口径下稳定高于：

* 行为策略 baseline（rule-based）
* 历史 DiscreteBC 基线

具体数值会随数据与配置变化，当前一版完整实验的结果可直接查看 `reports/summary_table.csv` 和 `artifacts/pipeline_eval/*.summary.json`。若 FQE/CPE 结果未明显优于上述两个基线，可调大训练步数或 conservative 相关超参；所有对比必须保持 reward_scale 一致，避免因尺度不同而误判效果。

千人千面分析 & 可视化
----------------------

离线分析脚本会读取 ReplayBuffer、user_profiles meta 以及训练好的策略，输出 CSV / PNG / JSON，供业务/前端进一步消费：

```bash
# 分群指标 + 柱状图
python -m hybrid_advisor_offline.offline.analysis.segment_metrics \
  --dataset ./data/offline_dataset_reward_personal.h5 \
  --behavior-meta ./data/offline_dataset_reward_personal_behavior.npz \
  --model ./models/bcq_reward_personal.pt \
  --algo-name bcq_reward_personal \
  --output-csv ./reports/segment_metrics_bcq_reward_personal.csv \
  --output-fig ./reports/segment_metrics_bcq_reward_personal.png

# 规则 / BC / BCQ 推荐差异样本
python -m hybrid_advisor_offline.offline.analysis.policy_diff \
  --dataset ./data/offline_dataset_reward_personal.h5 \
  --behavior-meta ./data/offline_dataset_reward_personal_behavior.npz \
  --bc-model ./models/bc_reward_personal.pt \
  --bcq-model ./models/bcq_reward_personal.pt \
  --num-scenarios 50 \
  --output ./reports/policy_diff_cases_reward_personal.json
```

生成的结果会写入 `./reports/`，Streamlit Demo 中新增的“千人千面分析”页签可直接读取这些文件：一侧展示分群 DataFrame + 柱状图，另一侧聚合规则 / BC / BCQ 的差异样本，方便在演示或答辩中快速对照。若目录下暂未生成这些文件，页面会提示相应命令。 
