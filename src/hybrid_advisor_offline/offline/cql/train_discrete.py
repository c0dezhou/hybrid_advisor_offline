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
N_STEPS = 50000 # 训练步数

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
    rew_stand = StandardRewardScaler()
    rew_stand.fit_with_transition_picker(dataset.episodes, dataset.transition_picker)
    return obs_stand,rew_stand

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
    )
    device =0 if require_gpu else False
    cql = config.create(device=device)

    print(f"{dataset.size()} 条轨迹上训练 DiscreteCQL，共 {N_STEPS} 步")
    cql.fit(
        train_buff,
        n_steps=N_STEPS,
        n_steps_per_epoch=1000,
        logger_adapter=FileAdapterFactory(root_dir="d3rlpy_logs/cql"),
        show_progress=True,
    )

    if not os.path.exists("./models"):
        os.makedirs("./models")
    cql.save_model(MODEL_SAVE_PATH)
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
    return parser.parse_args()


def main():
    args = _parse_args()

    global N_STEPS
    if args.steps is not None:
        N_STEPS = args.steps
        print(f"使用自定义训练步数 N_STEPS={N_STEPS}")

    try:
        tarin_discrete_cql(require_gpu=args.require_gpu)
    except RuntimeError as exc:
        if args.require_gpu:
            print(f"训练因 GPU 校验失败而退出：{exc}", file=sys.stderr)
        else:
            raise


if __name__ == "__main__":
    main()
