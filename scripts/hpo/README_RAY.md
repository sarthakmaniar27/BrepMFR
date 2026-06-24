# Ray Tune deferral note

Per project planning: **[Ray Tune](https://docs.ray.io/en/latest/tune/index.html)** is most valuable when you need **many parallel trials** across GPUs/machines. On a single Windows workstation the subprocess Optuna driver (`scripts/hpo/optuna_subprocess_stage1.py`) is simpler.

If you later move to Ray:

1. Wrap Lightning training in `train_loop_per_worker` ([Ray Train + Lightning](https://docs.ray.io/en/latest/train/getting-started-pytorch-lightning.html)).
2. Define `param_space` with `tune.choice`, `tune.loguniform`, etc., and use `Tuner.fit()` ([Tune basics](https://docs.ray.io/en/latest/tune/index.html)).
3. Report the monitored scalar (`eval_loss`) via `session.report(...)` or rely on Lightning callbacks Ray Tune integrates with.

Until parallel clusters are available, prefer Optuna + TensorBoard / CSV scalar scraping documented in [`README_OPTUNA.md`](README_OPTUNA.md).
