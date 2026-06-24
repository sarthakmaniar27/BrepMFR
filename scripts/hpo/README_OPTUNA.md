# Optuna + BrepMFR PyG

## Subprocess sweep (minimal coupling)

[`optuna_subprocess_stage1.py`](optuna_subprocess_stage1.py) launches `segmentation.py` per trial and reads back **`eval_loss`** from TensorBoard via [`tb_metrics.py`](tb_metrics.py). Extend the suggested hyperparameters inside `objective()` once you expose knobs on the CLI (e.g. learning rate, dropout).

Shared utilities:

- [`tb_metrics.py`](tb_metrics.py) — locate `tensorboard/version_*` under a run folder and read the latest scalar for candidate tags (`eval_loss`, `val/eval_loss`, …).

## Lightning pruning (tighter integration)

To prune bad trials mid-training, use Optuna’s
[`PyTorchLightningPruningCallback`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.PyTorchLightningPruningCallback.html):

```python
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial):
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="eval_loss")
    trainer = Trainer(
        callbacks=[pruning_callback, ...],
        ...
    )
    trainer.fit(...)
    return trainer.callback_metrics["eval_loss"].item()

study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
```

This requires refactoring `segmentation.py` / `domain_adapt.py` so training runs inside a callable `train_once(trial)` (or merging pruning callbacks into the existing Trainer constructed there).

Install:

```bash
pip install optuna
```

Dashboard:

```bash
optuna-dashboard sqlite:///study.db   # optional, persist study first
```
