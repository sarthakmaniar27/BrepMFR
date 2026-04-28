"""Print per-epoch Stage 2 metrics from the TensorBoard event file."""
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    args = ap.parse_args()

    ea = EventAccumulator(args.event, size_guidance={"scalars": 0})
    ea.Reload()

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
    ]
    data = {k: [(e.step, e.value) for e in ea.Scalars(k)] for k in keys}
    n = max(len(v) for v in data.values())

    header = (
        f"{'ep':>2}  {'tgt_acc':>7}  {'tgt_feat':>8}  {'src_acc':>7}  "
        f"{'L_s':>6}  {'L_t':>6}  {'L_adv':>6}  {'d_acc':>5}  {'lam':>5}  {'lr':>9}"
    )
    print(header)
    print("-" * len(header))

    def fmt(v, w, p):
        return f"{v:>{w}.{p}f}" if v is not None else " " * w

    for i in range(n):
        row = {}
        for k in keys:
            row[k] = data[k][i][1] if i < len(data[k]) else None
        print(
            f"{i:>2}  "
            f"{fmt(row['per_face_accuracy_target'], 7, 4)}  "
            f"{fmt(row['per_face_accuracy_target_feature'], 8, 4)}  "
            f"{fmt(row['per_face_accuracy_source'], 7, 4)}  "
            f"{fmt(row['train_loss_s'], 6, 4)}  "
            f"{fmt(row['train_loss_t'], 6, 4)}  "
            f"{fmt(row['train_loss_transfer'], 6, 4)}  "
            f"{fmt(row['train_transfer_acc'], 5, 1)}  "
            f"{fmt(row['grl_lambda'], 5, 3)}  "
            f"{fmt(row['current_lr'], 9, 6)}"
        )


if __name__ == "__main__":
    main()
