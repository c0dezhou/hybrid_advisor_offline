import os
import sys
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.dataset import ReplayBuffer, create_infinite_replay_buffer
from d3rlpy.dataset.buffers import InfiniteBuffer
from d3rlpy.preprocessing.observation_scalers import StandardObservationScaler
from d3rlpy.preprocessing.reward_scalers import StandardRewardScaler
from d3rlpy.logging import FileAdapterFactory
from sklearn.model_selection import train_test_split

from hybrid_advisor_offline.engine.act_safety.act_discrete_2_cards import get_act_space_size

DATASET_PATH = "./data/offline_dataset.h5"
MODEL_SAVE_PATH = "./models/cql_discrete_model.pt"
MODEL_CONFIG_PATH = f"{MODEL_SAVE_PATH}.config.json"
N_STEPS = 500000  # 训练步数
N_STEPS_PER_EPOCH = int(os.getenv("CQL_STEPS_PER_EPOCH", "5000"))
# 默认让 CQL 更加“大胆”：降低 alpha，减小学习率以避免不稳定
ALPHA = float(os.getenv("CQL_ALPHA", "0.01"))
LEARNING_RATE = float(os.getenv("CQL_LR", "2e-4"))
N_CRITICS = int(os.getenv("CQL_N_CRITICS", "2"))
TARGET_UPDATE_INTERVAL = int(os.getenv("CQL_TARGET_UPDATE", "8000"))
USE_REWARD_SCALER = os.getenv("CQL_USE_REWARD_SCALER", "1") != "0"

def _require_gpu():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
        "需要pytorch和GPU,请检查环境"
    ) from exc

    if not torch.cuda.is_available():
        print(
            "未检测到可用 GPU。训练已终止。如需在 CPU 上运行，请重新执行命令并移除 --require-gpu 参数。",
            file=sys.stderr,
        )
        raise RuntimeError(
            "需要GPU，请检查环境"
        )
    device_idx = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_idx)
    print(f"已检测到GPU:{device_name}(设备号：{device_idx})")


def _standard_dataset(dataset: ReplayBuffer):
    """
    将数据集标准化，主要是对state和reward（归一化到均值为0，方差为1）
    """
    obs_stand = StandardObservationScaler()
    obs_stand.fit_with_transition_picker(dataset.episodes, dataset.transition_picker)
    if USE_REWARD_SCALER:
        rew_stand = StandardRewardScaler()
        rew_stand.fit_with_transition_picker(dataset.episodes, dataset.transition_picker)
    else:
        rew_stand = None
    return obs_stand, rew_stand

def tarin_discrete_cql(require_gpu: bool):
    """
    使用离线数据集训练离散Conservative Qleaning
    """
    print("-------开始离散CQL训练")

    if require_gpu:
        _require_gpu()

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"未找到离线数据集：{DATASET_PATH}，请 gen offline dataset"
        )
    dataset = ReplayBuffer.load(DATASET_PATH, buffer=InfiniteBuffer())
    obs_stand, rew_stand = _standard_dataset(dataset)

    episo = list(dataset.episodes)
    train_episo, test_episo = train_test_split(episo, test_size=0.2, random_state=42)
    train_buff = create_infinite_replay_buffer(episodes=train_episo)
    test_buff = create_infinite_replay_buffer(episodes=test_episo)
    print(f"训练集轨迹总数{train_buff.size()},验证集轨迹总数：{test_buff.size()}")

    action_size = get_act_space_size()
    print(f"动作空间size:{action_size}")

    config = DiscreteCQLConfig(
        observation_scaler=obs_stand,
        reward_scaler=rew_stand,
        alpha=ALPHA,
        learning_rate=LEARNING_RATE,
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
    )
    device =0 if require_gpu else False
    cql = config.create(device=device)

    print(
        f"{dataset.size()} 条轨迹上训练 DiscreteCQL，共 {N_STEPS} 步 "
        f"(alpha={ALPHA}, lr={LEARNING_RATE}, n_critics={N_CRITICS}, "
        f"steps/epoch={N_STEPS_PER_EPOCH}, reward_scaler={'on' if USE_REWARD_SCALER else 'off'})"
    )
    cql.fit(
        train_buff,
        n_steps=N_STEPS,
        n_steps_per_epoch=N_STEPS_PER_EPOCH,
        logger_adapter=FileAdapterFactory(root_dir="d3rlpy_logs/cql"),
        show_progress=True,
    )

    if not os.path.exists("./models"):
        os.makedirs("./models")
    cql.save_model(MODEL_SAVE_PATH)
    config_payload = {
        "alpha": ALPHA,
        "learning_rate": LEARNING_RATE,
        "n_critics": N_CRITICS,
        "target_update_interval": TARGET_UPDATE_INTERVAL,
        "n_steps_per_epoch": N_STEPS_PER_EPOCH,
        "use_reward_scaler": USE_REWARD_SCALER,
    }
    try:
        import json

        with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as cfg_file:
            json.dump(config_payload, cfg_file, ensure_ascii=False, indent=2)
    except OSError as exc:
        print(f"警告：写入模型配置 {MODEL_CONFIG_PATH} 失败：{exc}", file=sys.stderr)
    else:
        print(f"训练配置已保存至 {MODEL_CONFIG_PATH}")
    print(f"\n训练完成，模型已保存至 {MODEL_SAVE_PATH}")

def load_cql_policy(require_gpu: bool = True) -> DiscreteCQL:
    """
    加载已经训练好的离散 CQL 策略。
    """
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"未在 {MODEL_SAVE_PATH} 找到已训练的模型请先 train_discrete_cql.py"
        )

    if require_gpu:
        _require_gpu()

    # act_size = get_act_space_size()
    dataset = ReplayBuffer.load(DATASET_PATH, buffer=InfiniteBuffer())
    obs_scaler, rew_scaler = _standard_dataset(dataset)
    config = DiscreteCQLConfig(
        observation_scaler=obs_scaler,
        reward_scaler=rew_scaler,
        alpha=ALPHA,
        learning_rate=LEARNING_RATE,
        n_critics=N_CRITICS,
        target_update_interval=TARGET_UPDATE_INTERVAL,
    )
    device = 0 if require_gpu else False
    policy = config.create(device=device)
    policy.build_with_dataset(dataset)
    policy.load_model(MODEL_SAVE_PATH)

    print(f"已从 {MODEL_SAVE_PATH} 成功加载 CQL 策略")
    return policy    


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="使用离线数据集训练离散 Conservative Q-Learning 模型。",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="若指定，则检测并强制使用 GPU；若未检测到 GPU，则终止训练。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"自定义训练步数（默认使用 N_STEPS={N_STEPS}）。用于调试时可减小。",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help=f"每个 epoch 内的更新步数（默认 {N_STEPS_PER_EPOCH}）。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=f"CQL 保守系数（默认 {ALPHA}，数值越大越保守）。",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"优化器学习率（默认 {LEARNING_RATE}）。",
    )
    parser.add_argument(
        "--n-critics",
        type=int,
        default=None,
        help=f"Critic 数量（默认 {N_CRITICS}）。",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=None,
        help=f"Target 网络更新步间隔（默认 {TARGET_UPDATE_INTERVAL}）。",
    )
    parser.add_argument(
        "--no-reward-scaler",
        action="store_true",
        help="禁用 reward scaler，直接使用原始奖励值。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    global N_STEPS, N_STEPS_PER_EPOCH, ALPHA, LEARNING_RATE, N_CRITICS, TARGET_UPDATE_INTERVAL, USE_REWARD_SCALER
    if args.steps is not None:
        N_STEPS = args.steps
        print(f"使用自定义训练步数 N_STEPS={N_STEPS}")
    if args.steps_per_epoch is not None:
        global N_STEPS_PER_EPOCH
        N_STEPS_PER_EPOCH = args.steps_per_epoch
        print(f"使用自定义 steps_per_epoch={N_STEPS_PER_EPOCH}")
    if args.alpha is not None:
        global ALPHA
        ALPHA = args.alpha
        print(f"使用自定义 alpha={ALPHA}")
    if args.learning_rate is not None:
        global LEARNING_RATE
        LEARNING_RATE = args.learning_rate
        print(f"使用自定义 learning_rate={LEARNING_RATE}")
    if args.n_critics is not None:
        global N_CRITICS
        N_CRITICS = args.n_critics
        print(f"使用自定义 n_critics={N_CRITICS}")
    if args.target_update_interval is not None:
        global TARGET_UPDATE_INTERVAL
        TARGET_UPDATE_INTERVAL = args.target_update_interval
        print(f"使用自定义 target_update_interval={TARGET_UPDATE_INTERVAL}")
    if args.no_reward_scaler:
        global USE_REWARD_SCALER
        USE_REWARD_SCALER = False
        print("已禁用 reward scaler，训练将使用原始奖励值。")

    try:
        tarin_discrete_cql(require_gpu=args.require_gpu)
    except RuntimeError as exc:
        if args.require_gpu:
            print(f"训练因 GPU 校验失败而退出：{exc}", file=sys.stderr)
        else:
            raise


if __name__ == "__main__":
    main()


"""
loss越来越大了，说明保守项在强行拉回策略，进一步下调 alpha（例如 0.02 甚至 0.01），减轻 conservative 项，调低lr至 3e-4 或 2e-4
Epoch 11/100: 100%|████████████████| 8000/8000 [01:27<00:00, 91.15it/s, loss=2.38, td_loss=2.23, conservative_loss=2.95]
2025-11-07 20:50.44 [info     ] DiscreteCQL_20251107203504: epoch=11 step=88000 epoch=11 metrics={'time_sample_batch': 0
.0003294392228126526, 'time_algorithm_update': 0.010565196573734283, 'loss': 2.3805395580269395, 'td_loss': 2.2329370494
629255, 'conservative_loss': 2.9520501913130284, 'time_step': 0.010950414955615998} step=88000
2025-11-07 20:50.44 [info     ] Model parameters are saved to d3rlpy_logs/cql/DiscreteCQL_20251107203504/model_88000.d3
Epoch 12/100:   8%|█▎               | 641/8000 [00:06<01:19, 92.48it/s, loss=2.18, td_loss=2.03, conservative_loss=2.91]
Epoch 12/100:   8%|█▎               | 644/8000 [00:06<01:19, 92.36it/s, loss=2.18, td_loss=2.03, conservative_loss=2.91]
"""