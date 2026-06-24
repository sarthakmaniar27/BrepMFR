# -*- coding: utf-8 -*-
"""Dump scalar summaries from a TensorBoard logdir (EventAccumulator).

Also inspect ``meta/*`` tags emitted by ``TrainingMetaLoggerCallback``
(dataset lengths, class-weight alpha, etc.).
"""
import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "logdir",
        help="Folder containing events.out.tfevents* (usually .../tensorboard/version_*/ after the logs split).",
    )
    args = ap.parse_args()
    root = Path(args.logdir)
    events = list(root.rglob("events.out.tfevents*"))
    if not events:
        print(f"No event files under {root.resolve()}")
        return
    parent = events[0].parent
    ea = EventAccumulator(str(parent), size_guidance={"scalars": 0})
    ea.Reload()
    tags = sorted(ea.Tags().get("scalars", []))
    print(f"logdir: {parent.resolve()}")
    print(f"event files: {len(events)}")
    print(f"scalar tags: {len(tags)}")
    for t in tags:
        s = ea.Scalars(t)
        if not s:
            continue
        vals = [x.value for x in s]
        steps = [x.step for x in s]
        print(
            f"{t}: n={len(s)} step_last={steps[-1]} "
            f"last={vals[-1]:.8g} first={vals[0]:.8g} min={min(vals):.8g} max={max(vals):.8g}"
        )


if __name__ == "__main__":
    main()
