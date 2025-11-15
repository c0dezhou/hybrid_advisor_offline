import argparse
from pathlib import Path

from hybrid_advisor_offline.offline.eval.policy_loader import (
    load_policy_artifact,
)


def export_single(model_path: Path, output_path: Path, require_gpu: bool = False) -> None:
    """
    使用现有的完整加载路径恢复策略，然后将其以轻量形式保存，
    供前端 / 推理服务直接使用。
    """
    model_path = model_path.resolve()
    output_path = output_path.resolve()
    print(f"[export] loading full policy from {model_path}")
    policy = load_policy_artifact(str(model_path), require_gpu=require_gpu)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] saving inference artifact to {output_path}")
    policy.save(str(output_path))
    print("[export] done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将训练好的策略模型导出为推理专用文件（含 scaler），"
        "前端可通过 load_policy_inference 进行轻量加载。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="训练阶段生成的 .pt 模型路径（例如 ./models/bcq_reward_personal.pt）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="推理专用文件输出路径，默认为在原路径旁边加后缀 .inference.d3",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="在有 GPU 时，用 GPU 设备加载后再导出（通常可不指定）。",
    )
    args = parser.parse_args()

    model_path: Path = args.model
    if args.output is None:
        output = model_path.with_suffix(model_path.suffix + ".inference.d3")
    else:
        output = args.output

    export_single(model_path, output, require_gpu=args.use_gpu)


if __name__ == "__main__":
    main()

