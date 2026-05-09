"""
Durable 1-hour Stage 2 Run 4 TensorBoard + terminal poller.

Each tick:
  - Reads all scalars from the run folder via tensorboard.event_accumulator.
  - Prints per-epoch deltas since the last tick for key scalars.
  - Tails the last ~80 lines of the training terminal file.
  - Flags warning conditions.

The output is written to this process's stdout, which the Shell tool captures
into terminals/<shell_id>.txt so we can read it on demand.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator as ea

RUN_DIR = Path(r"C:\Users\D58\Desktop\BrepMFR\results\BrepToSeq-segmentation\0421\152709")
TRAIN_TERMINAL_FILE = Path(
    r"C:\Users\D58\.cursor\projects\c-Users-D58-Desktop-BrepMFR\terminals\254093.txt"
)
INTERVAL_S = 3600
TAIL_LINES = 80

KEY_SCALARS = [
    "current_lr",
    "grl_lambda",
    "train_loss_s",
    "train_loss_t",
    "train_loss_transfer",
    "train_transfer_acc",
    "train_acc_s_epoch",
    "train_acc_t_epoch",
    "eval_loss_s",
    "eval_loss_t",
    "eval_loss_transfer",
    "per_face_accuracy_source",
    "per_face_accuracy_target",
    "per_face_accuracy_target_feature",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    a = ea.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        a.Reload()
    except Exception as exc:
        print(f"[poll {now_utc()}] tensorboard reload failed: {exc!r}")
        return {}
    tags = a.Tags().get("scalars", [])
    out: dict[str, list[tuple[int, float]]] = {}
    for t in tags:
        try:
            es = a.Scalars(t)
            out[t] = [(e.step, float(e.value)) for e in es]
        except Exception as exc:
            print(f"[poll {now_utc()}] scalar read failed for {t}: {exc!r}")
    return out


def last_value(series: list[tuple[int, float]]) -> tuple[int | None, float | None]:
    if not series:
        return None, None
    s, v = series[-1]
    return s, v


def tail_file(path: Path, nlines: int) -> list[str]:
    if not path.exists():
        return [f"[tail] file not found: {path}"]
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as exc:
        return [f"[tail] read failed: {exc!r}"]
    return [ln.rstrip() for ln in lines[-nlines:]]


def alerts(data: dict[str, list[tuple[int, float]]], best_tgt: dict[str, float]) -> list[str]:
    msgs: list[str] = []
    # loss_s blowup
    _, loss_s_last = last_value(data.get("train_loss_s", []))
    if loss_s_last is not None and loss_s_last > 0.05:
        msgs.append(f"[ALERT] train_loss_s={loss_s_last:.4f} > 0.05 (A0 may be too aggressive; consider rollback to alpha=5)")
    # B1 entropy collapse (train_loss_t < 0.005 was the Run 2 failure signature at entropy=0.1)
    _, loss_t_last = last_value(data.get("train_loss_t", []))
    if loss_t_last is not None and loss_t_last < 0.005:
        msgs.append(f"[ALERT] train_loss_t={loss_t_last:.4f} < 0.005 (entropy B1 collapsing; reduce coefficient below 0.02)")
    # per_face_accuracy_target vs rolling best
    tgt = data.get("per_face_accuracy_target", [])
    if tgt:
        cur = tgt[-1][1]
        bst = max(v for _, v in tgt)
        best_tgt["best"] = max(best_tgt.get("best", 0.0), bst)
        if cur + 0.03 < best_tgt["best"]:
            msgs.append(f"[ALERT] per_face_accuracy_target={cur:.4f} dropped >3pp below rolling best {best_tgt['best']:.4f}")
    # train_transfer_acc trend (should fall; flag if rising after ep 3)
    ttc = data.get("train_transfer_acc", [])
    if len(ttc) >= 5:
        e_now = len(ttc)
        if e_now >= 3:
            early = ttc[2][1]
            latest = ttc[-1][1]
            if latest > early + 3.0:  # percentage points
                msgs.append(f"[ALERT] train_transfer_acc rose from {early:.2f} (ep3) to {latest:.2f} (ep{e_now}) — disc winning, not losing")
    return msgs


def format_tick(tick_num: int, data: dict[str, list[tuple[int, float]]], prev_ep: dict[str, int]) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append(f"[tick {tick_num}] {now_utc()}  run_dir={RUN_DIR}")
    lines.append("=" * 100)

    # dataset sizes & basic progress
    epoch_series = data.get("epoch", [])
    train_step_series = data.get("train_acc_s_step", [])
    if epoch_series:
        last_epoch = int(epoch_series[-1][1])
        total_ticks = len(epoch_series)
        lines.append(f"epoch tag last_step={epoch_series[-1][0]}  last_epoch_value={last_epoch}  total_ticks={total_ticks}")
    if train_step_series:
        lines.append(f"train_acc_s_step last step={train_step_series[-1][0]}  val={train_step_series[-1][1]:.4f}")

    # per-epoch key scalars table
    lines.append("")
    lines.append("Per-epoch key scalars (last 8 epochs):")
    # align by index, not step
    N = max([len(data.get(t, [])) for t in KEY_SCALARS] + [0])
    if N == 0:
        lines.append("  (no epoch-level scalars yet — waiting for first epoch to complete)")
    else:
        hdr = "ep  lr        grl_l    loss_s    loss_t    loss_adv  disc%   tr_acc_s tr_acc_t ev_loss_s ev_adv  src_acc  tgt_acc  feat_acc"
        lines.append(hdr)
        start = max(0, N - 8)
        for i in range(start, N):
            def g(tag: str, idx: int = i) -> str:
                ser = data.get(tag, [])
                if idx < len(ser):
                    return f"{ser[idx][1]:.5f}"
                return "-"
            def gp(tag: str, idx: int = i) -> str:
                ser = data.get(tag, [])
                if idx < len(ser):
                    return f"{ser[idx][1]:.2f}"
                return "-"
            row = (
                f"{i+1:3d} "
                f"{g('current_lr'):<9} "
                f"{g('grl_lambda'):<8} "
                f"{g('train_loss_s'):<9} "
                f"{g('train_loss_t'):<9} "
                f"{g('train_loss_transfer'):<9} "
                f"{gp('train_transfer_acc'):<7} "
                f"{g('train_acc_s_epoch'):<8} "
                f"{g('train_acc_t_epoch'):<8} "
                f"{g('eval_loss_s'):<9} "
                f"{g('eval_loss_transfer'):<7} "
                f"{g('per_face_accuracy_source'):<8} "
                f"{g('per_face_accuracy_target'):<8} "
                f"{g('per_face_accuracy_target_feature'):<8}"
            )
            lines.append(row)
    # delta since last tick
    lines.append("")
    lines.append("Delta since last tick:")
    for t in KEY_SCALARS:
        ser = data.get(t, [])
        if not ser:
            continue
        cur_ep = len(ser)
        prev = prev_ep.get(t, -1)
        _, last_v = ser[-1]
        if prev >= 0 and prev < cur_ep:
            lines.append(f"  {t}: ep{prev}->ep{cur_ep}  last_val={last_v:.5f}")
        else:
            lines.append(f"  {t}: ep{cur_ep}  last_val={last_v:.5f}")
        prev_ep[t] = cur_ep

    return "\n".join(lines)


def main() -> None:
    print(f"[poller] start {now_utc()} pid={os.getpid()}")
    print(f"[poller] run_dir     = {RUN_DIR}")
    print(f"[poller] train_term  = {TRAIN_TERMINAL_FILE}")
    print(f"[poller] interval_s  = {INTERVAL_S}")
    prev_ep: dict[str, int] = {}
    best_tgt: dict[str, float] = {}
    tick = 0
    while True:
        tick += 1
        try:
            data = read_scalars(RUN_DIR)
            block = format_tick(tick, data, prev_ep)
            print(block)
            for a in alerts(data, best_tgt):
                print(a)
            print("")
            print(f"--- tail({TAIL_LINES}) of training terminal ---")
            for ln in tail_file(TRAIN_TERMINAL_FILE, TAIL_LINES):
                print(ln)
            print("--- end tail ---")
        except Exception as exc:
            print(f"[poll {now_utc()}] tick failed: {exc!r}")
        sys.stdout.flush()
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
