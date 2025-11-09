import builtins
import sys
import types

import pytest

from hybrid_advisor_offline.offline.trainrl import train_discrete


def test_require_gpu_import_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        train_discrete._require_gpu()


def test_require_gpu_without_cuda(monkeypatch):
    dummy_cuda = types.SimpleNamespace(
        is_available=lambda: False,
        current_device=lambda: 0,
        get_device_name=lambda idx: "stub-device",
    )
    dummy_torch = types.SimpleNamespace(cuda=dummy_cuda)
    monkeypatch.setitem(sys.modules, "torch", dummy_torch)

    with pytest.raises(RuntimeError):
        train_discrete._require_gpu()


def test_tarin_discrete_cql_missing_dataset(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.h5"
    monkeypatch.setattr(train_discrete, "DATASET_PATH", str(missing_path))

    with pytest.raises(FileNotFoundError):
        train_discrete.tarin_discrete_cql(require_gpu=False)


def test_load_cql_policy_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(train_discrete, "MODEL_SAVE_PATH", str(tmp_path / "model.pt"))

    with pytest.raises(FileNotFoundError):
        train_discrete.load_cql_policy(require_gpu=False)


def test_load_cql_policy_builds(monkeypatch, tmp_path):
    dataset_path = tmp_path / "offline_dataset.h5"
    dataset_path.write_text("dataset")
    model_path = tmp_path / "saved_model.pt"
    model_path.write_text("trained")

    monkeypatch.setattr(train_discrete, "DATASET_PATH", str(dataset_path))
    monkeypatch.setattr(train_discrete, "MODEL_SAVE_PATH", str(model_path))

    class DummyDataset:
        episodes = []
        transition_picker = object()

        @staticmethod
        def size():
            return 0

    class DummyReplayBuffer:
        @classmethod
        def load(cls, path, *, buffer):
            return DummyDataset()

    class DummyPolicy:
        def build_with_dataset(self, dataset):
            self.dataset = dataset

        def load_model(self, path):
            self.path = path

    class DummyConfig:
        def __init__(self, **kwargs):
            pass

        def create(self, device):
            return DummyPolicy()

    monkeypatch.setattr(train_discrete, "_require_gpu", lambda: None)
    monkeypatch.setattr(train_discrete, "ReplayBuffer", DummyReplayBuffer)
    monkeypatch.setattr(train_discrete, "InfiniteBuffer", lambda: "buf")
    monkeypatch.setattr(train_discrete, "_standard_dataset", lambda dataset: ("obs", "rew"))
    monkeypatch.setattr(train_discrete, "DiscreteCQLConfig", DummyConfig)

    policy = train_discrete.load_cql_policy(require_gpu=False)
    assert isinstance(policy, DummyPolicy)
