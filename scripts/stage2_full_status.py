"""Dump every scalar tag in a Stage 2 tfevents file with step + value."""
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    args = ap.parse_args()

    ea = EventAccumulator(args.event, size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print(f"# scalar tags ({len(tags)}):")
    for t in tags:
        scalars = ea.Scalars(t)
        if not scalars:
            continue
        first, last = scalars[0], scalars[-1]
        print(f"  {t:<40s}  count={len(scalars):>5}  step_first={first.step:>7}  step_last={last.step:>7}  "
              f"first_val={first.value:.4f}  last_val={last.value:.4f}")

    # Also print full table for the standard per-epoch keys.
    keys = [
        "per_face_accuracy_target",
        "per_face_accuracy_target_feature",
        "per_face_accuracy_source",
        "train_loss_s",
        "train_loss_t",
        "train_loss_transfer",
        "train_transfer_acc",
        "grl_lambda",
        "current_lr",
        "eval_loss",
        "eval_loss_s",
        "eval_loss_t",
        "eval_loss_transfer",
    ]
    data = {k: [(e.step, e.value) for e in ea.Scalars(k)] if k in tags else [] for k in keys}
    n = max((len(v) for v in data.values()), default=0)
    if n == 0:
        return

    header = (
        f"\n{'ep':>2} {'step':>7}  {'tgt_acc':>7} {'tgt_feat':>8} {'src_acc':>7}  "
        f"{'L_s':>6} {'L_t':>6} {'L_adv':>6} {'d_acc':>5} {'lam':>6}  {'lr':>9}  "
        f"{'eval_loss':>9}"
    )
    print(header)
    print("-" * len(header))

    def fmt(v, w, p):
        return f"{v:>{w}.{p}f}" if v is not None else " " * w

    for i in range(n):
        row = {}
        step_val = None
        for k in keys:
            if i < len(data[k]):
                row[k] = data[k][i][1]
                if step_val is None:
                    step_val = data[k][i][0]
            else:
                row[k] = None
        print(
            f"{i:>2} {step_val if step_val is not None else 0:>7}  "
            f"{fmt(row['per_face_accuracy_target'], 7, 4)} "
            f"{fmt(row['per_face_accuracy_target_feature'], 8, 4)} "
            f"{fmt(row['per_face_accuracy_source'], 7, 4)}  "
            f"{fmt(row['train_loss_s'], 6, 4)} "
            f"{fmt(row['train_loss_t'], 6, 4)} "
            f"{fmt(row['train_loss_transfer'], 6, 4)} "
            f"{fmt(row['train_transfer_acc'], 5, 1)} "
            f"{fmt(row['grl_lambda'], 6, 4)}  "
            f"{fmt(row['current_lr'], 9, 6)}  "
            f"{fmt(row['eval_loss'], 9, 4)}"
        )


if __name__ == "__main__":
    main()
