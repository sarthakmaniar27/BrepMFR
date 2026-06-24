# -*- coding: utf-8 -*-
"""
Example Optuna study driving ``segmentation.py`` via subprocess.

Install: ``pip install optuna``

Minimizes the last logged validation ``eval_loss`` read back from TensorBoard.

Usage (from repo root)::

  python scripts/hpo/optuna_subprocess_stage1.py --dataset_path Z:/path/to/source_dataset --trials 12

Each trial writes ``results/stage1/optuna_s1_t<N>/``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--study_name", type=str, default="stage1_smoke_epochs")
    args_ns = ap.parse_args()

    try:
        import optuna
    except ImportError as e:
        raise SystemExit("Install optuna: pip install optuna") from e

    sys.path.insert(0, str(REPO_ROOT))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tb_metrics", REPO_ROOT / "scripts" / "hpo" / "tb_metrics.py"
    )
    assert spec and spec.loader
    tb_metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb_metrics)
    find_tensorboard_scalar_dir = tb_metrics.find_tensorboard_scalar_dir
    latest_scalar_value = tb_metrics.latest_scalar_value

    def objective(trial) -> float:
        max_epochs = trial.suggest_int("max_epochs", 4, 12)
        run_name = f"optuna_s1_t{trial.number}"
        run_dir = REPO_ROOT / "results" / "stage1" / run_name
        if run_dir.is_dir():
            shutil.rmtree(run_dir, ignore_errors=True)

        cmd = [
            sys.executable,
            str(REPO_ROOT / "segmentation.py"),
            "train",
            "--dataset_path",
            args_ns.dataset_path,
            "--num_classes",
            str(args_ns.num_classes),
            "--batch_size",
            str(args_ns.batch_size),
            "--num_workers",
            str(args_ns.num_workers),
            "--max_epochs",
            str(max_epochs),
            "--run_name",
            run_name,
        ]

        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

        scalar_dir = find_tensorboard_scalar_dir(run_dir)
        if scalar_dir is None:
            return 1e9
        loss = latest_scalar_value(
            scalar_dir,
            ["eval_loss", "val/eval_loss", "epoch_eval_loss"],
        )
        if loss is None:
            return 1e9
        return float(loss)

    study = optuna.create_study(study_name=args_ns.study_name, direction="minimize")
    study.optimize(objective, n_trials=args_ns.trials)
    print("Best:", study.best_params, study.best_value)


if __name__ == "__main__":
    main()
