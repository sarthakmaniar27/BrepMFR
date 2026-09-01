#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unsupervised_training.checkpointing import extract_student_checkpoint  # noqa: E402
from unsupervised_training.constants import REPO_ROOT  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the student and export validated A1+A3 ONNX for SolidWorks"
    )
    parser.add_argument("--joint-checkpoint", required=True)
    parser.add_argument("--champion-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-onnx", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "student_stage1.ckpt"
    extract_student_checkpoint(
        args.joint_checkpoint,
        args.champion_checkpoint,
        stage1_path,
    )
    print(f"Extracted deployable Stage-1 checkpoint: {stage1_path}")
    if args.skip_onnx:
        return

    exporter = REPO_ROOT / "migration_to_c++" / "migration_to_c" / "model_conversion_onnx.py"
    if not exporter.is_file():
        raise FileNotFoundError(f"A1+A3 ONNX exporter not found: {exporter}")
    command = [
        sys.executable,
        str(exporter),
        "--checkpoint",
        str(stage1_path),
        "--output_dir",
        str(output_dir),
    ]
    if args.skip_validation:
        command.append("--skip_validation")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()

