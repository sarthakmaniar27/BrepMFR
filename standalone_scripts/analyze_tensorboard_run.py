#!/usr/bin/env python3
"""Generate a manager-friendly HTML/Markdown report from TensorBoard logs.

The report focuses on Stage-1 face segmentation metrics:

* weighted train/validation loss;
* overall face accuracy, macro class accuracy, feature-only accuracy, and mIoU;
* per-class recall and IoU;
* learning-rate behavior and A1/A3 contribution;
* convergence, class-balance, and possible overfitting observations.

Example:

    python scripts/training/analyze_tensorboard_run.py \
      --run-name five_class_a1_a3_scratch_20260806_214444

Or point directly at a TensorBoard directory:

    python scripts/training/analyze_tensorboard_run.py \
      --log-dir results/logs/stage1/my_run/tensorboard \
      --output-dir results/reports/my_run
"""

from __future__ import annotations

import argparse
import base64
import bisect
import csv
import html
import io
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "TensorBoard reader is unavailable. Install it with: pip install tensorboard"
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "Matplotlib is unavailable. Install it with: pip install matplotlib"
    ) from exc


DEFAULT_CLASS_NAMES = ["Stock", "Thread", "Text", "Chamfer", "Fillet"]
QUALITY_TAGS = {
    "Macro class accuracy": "per_class_accuracy",
    "Overall face accuracy": "per_face_accuracy",
    "Feature-only accuracy": "per_face_accuracy_feature",
    "Mean IoU": "IoU",
}


@dataclass(frozen=True)
class Point:
    step: int
    epoch: float
    value: float
    wall_time: float


def _event_directories(root: Path) -> list[Path]:
    if root.is_file() and root.name.startswith("events.out.tfevents"):
        return [root.parent]
    if not root.is_dir():
        raise FileNotFoundError(f"TensorBoard path does not exist: {root}")
    direct = list(root.glob("events.out.tfevents.*"))
    if direct:
        return [root]
    directories = sorted(
        {
            path.parent
            for path in root.rglob("events.out.tfevents.*")
            if path.is_file()
        }
    )
    if not directories:
        raise FileNotFoundError(f"No TensorBoard event files found under: {root}")
    return directories


def _resolve_log_root(repo_root: Path, run_name: Optional[str], log_dir: Optional[Path]) -> Path:
    if log_dir is not None:
        return log_dir.expanduser().resolve()
    if not run_name:
        raise ValueError("Specify either --run-name or --log-dir")
    candidates = [
        repo_root / "results" / "logs" / "stage1" / run_name / "tensorboard",
        repo_root / "results" / "stage1" / run_name / "tensorboard",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not locate TensorBoard logs for run {run_name!r}. Tried:\n{attempted}"
    )


def _load_raw_scalars(event_dirs: Iterable[Path]) -> dict[str, list]:
    by_tag: dict[str, list] = {}
    for directory in event_dirs:
        accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            by_tag.setdefault(tag, []).extend(accumulator.Scalars(tag))
    if not by_tag:
        raise RuntimeError("TensorBoard event files contain no scalar metrics.")
    return by_tag


def _dedupe_raw(events: Iterable) -> list:
    """Keep the newest write for each step, which handles interrupted-epoch resumes."""
    newest = {}
    for event in events:
        key = int(event.step)
        previous = newest.get(key)
        if previous is None or float(event.wall_time) >= float(previous.wall_time):
            newest[key] = event
    return [newest[step] for step in sorted(newest)]


def _build_epoch_lookup(raw: dict[str, list]) -> tuple[list[int], list[float]]:
    epoch_events = _dedupe_raw(raw.get("epoch", []))
    return (
        [int(event.step) for event in epoch_events],
        [float(event.value) for event in epoch_events],
    )


def _epoch_for_step(step: int, epoch_steps: list[int], epoch_values: list[float]) -> float:
    if not epoch_steps:
        return float(step)
    index = bisect.bisect_right(epoch_steps, step) - 1
    if index < 0:
        return float(epoch_values[0])
    return float(epoch_values[index])


def _normalise_scalars(raw: dict[str, list]) -> dict[str, list[Point]]:
    epoch_steps, epoch_values = _build_epoch_lookup(raw)
    output: dict[str, list[Point]] = {}
    for tag, raw_events in raw.items():
        points: list[Point] = []
        for event in _dedupe_raw(raw_events):
            # Media helpers intentionally log validation class metrics with
            # current_epoch as the TensorBoard step. Lightning metrics use
            # optimizer global_step and need the epoch lookup.
            epoch = (
                float(event.step)
                if tag.startswith("val/per_class_")
                else _epoch_for_step(int(event.step), epoch_steps, epoch_values)
            )
            points.append(
                Point(
                    step=int(event.step),
                    epoch=epoch,
                    value=float(event.value),
                    wall_time=float(event.wall_time),
                )
            )
        output[tag] = points
    return output


def _first_series(series: dict[str, list[Point]], *tags: str) -> list[Point]:
    for tag in tags:
        if series.get(tag):
            return series[tag]
    return []


def _finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def _fmt(value: Optional[float], percent: bool = False) -> str:
    if not _finite(value):
        return "not logged"
    if percent:
        return f"{100.0 * float(value):.1f}%"
    return f"{float(value):.4f}"


def _latest(points: list[Point]) -> Optional[Point]:
    return points[-1] if points else None


def _best(points: list[Point], maximise: bool = True) -> Optional[Point]:
    if not points:
        return None
    key = (lambda point: point.value) if maximise else (lambda point: -point.value)
    return max(points, key=key)


def _recent_delta(points: list[Point], count: int = 3) -> Optional[float]:
    if len(points) < 2:
        return None
    window = points[-min(count, len(points)) :]
    return float(window[-1].value - window[0].value)


def _latest_at_or_before(points: list[Point], epoch: float) -> Optional[Point]:
    candidates = [point for point in points if point.epoch <= epoch + 1e-6]
    return candidates[-1] if candidates else _latest(points)


def _class_series(
    series: dict[str, list[Point]],
    class_index: int,
    metric: str,
) -> list[Point]:
    if metric == "recall":
        return _first_series(
            series,
            f"val/per_class_recall/c{class_index:02d}",
            f"val_class_{class_index}_acc",
        )
    return _first_series(series, f"val/per_class_iou/c{class_index:02d}")


def _metric_summary(series: dict[str, list[Point]]) -> dict:
    macro = _first_series(series, "per_class_accuracy")
    overall = _first_series(series, "per_face_accuracy")
    feature = _first_series(series, "per_face_accuracy_feature")
    iou = _first_series(series, "IoU")
    train_loss = _first_series(series, "train_loss_epoch", "train_loss")
    eval_loss = _first_series(series, "eval_loss")
    current_lr = _first_series(series, "current_lr")
    a1_a3_lr = _first_series(series, "a1_a3_lr")
    a1_a3_scale = _first_series(series, "a1_a3_scale")

    selection_name = "Macro class accuracy"
    selection = macro
    maximise = True
    if not selection:
        selection_name, selection = "Mean IoU", iou
    if not selection:
        selection_name, selection, maximise = "Validation loss", eval_loss, False

    best_selection = _best(selection, maximise=maximise)
    latest_selection = _latest(selection)
    all_points = [point for points in series.values() for point in points]
    start_time = min((point.wall_time for point in all_points), default=0.0)
    end_time = max((point.wall_time for point in all_points), default=0.0)

    return {
        "selection_name": selection_name,
        "selection_maximise": maximise,
        "selection": selection,
        "best_selection": best_selection,
        "latest_selection": latest_selection,
        "macro": macro,
        "overall": overall,
        "feature": feature,
        "iou": iou,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "current_lr": current_lr,
        "a1_a3_lr": a1_a3_lr,
        "a1_a3_scale": a1_a3_scale,
        "start_time": start_time,
        "end_time": end_time,
    }


def _interpret(
    summary: dict,
    series: dict[str, list[Point]],
    class_names: list[str],
) -> tuple[str, list[str], list[dict]]:
    observations: list[str] = []
    best = summary["best_selection"]
    latest = summary["latest_selection"]
    selection = summary["selection"]
    recent_change = _recent_delta(selection)
    if not summary["selection_maximise"] and recent_change is not None:
        recent_change *= -1.0

    status = "Insufficient validation history"
    if len(selection) >= 2:
        if recent_change is not None and recent_change > 0.01:
            status = "Still improving"
            observations.append(
                f"{summary['selection_name']} is still improving over the latest "
                f"{min(3, len(selection))} validation points."
            )
        elif recent_change is not None and recent_change < -0.02:
            status = "Recent validation decline"
            observations.append(
                f"{summary['selection_name']} declined recently; use the best checkpoint "
                "rather than the last checkpoint for evaluation."
            )
        else:
            status = "Near a plateau"
            observations.append(
                f"{summary['selection_name']} has changed little over the latest "
                "validation points, suggesting convergence or a plateau."
            )

    if best and latest and summary["selection_maximise"]:
        drop = best.value - latest.value
        if drop > 0.02:
            observations.append(
                f"Latest {summary['selection_name'].lower()} is {_fmt(drop, percent=True)} "
                "below its best value, a possible sign of overfitting or training variance."
            )

    train_latest = _latest(summary["train_loss"])
    eval_latest = _latest(summary["eval_loss"])
    eval_best = _best(summary["eval_loss"], maximise=False)
    if train_latest and eval_latest and eval_best:
        if eval_latest.value > eval_best.value * 1.10 and eval_latest.epoch > eval_best.epoch:
            observations.append(
                "Training loss remains low while validation loss has moved more than 10% "
                "above its minimum; this is an overfitting warning."
            )

    overall_latest = _latest(summary["overall"])
    macro_latest = _latest(summary["macro"])
    if overall_latest and macro_latest:
        gap = overall_latest.value - macro_latest.value
        if gap > 0.05:
            observations.append(
                f"Overall face accuracy exceeds macro class accuracy by "
                f"{_fmt(gap, percent=True)}. Frequent classes are performing better than "
                "the average class, so overall accuracy alone is optimistic."
            )

    class_rows: list[dict] = []
    target_epoch = latest.epoch if latest else float("inf")
    for index, name in enumerate(class_names):
        recall = _latest_at_or_before(_class_series(series, index, "recall"), target_epoch)
        class_iou = _latest_at_or_before(_class_series(series, index, "iou"), target_epoch)
        class_rows.append(
            {
                "index": index,
                "name": name,
                "recall": recall.value if recall else None,
                "iou": class_iou.value if class_iou else None,
            }
        )

    rows_with_recall = [row for row in class_rows if _finite(row["recall"])]
    if rows_with_recall:
        weakest = min(rows_with_recall, key=lambda row: row["recall"])
        strongest = max(rows_with_recall, key=lambda row: row["recall"])
        gap = strongest["recall"] - weakest["recall"]
        observations.append(
            f"Weakest class by validation recall is {weakest['name']} "
            f"({_fmt(weakest['recall'], percent=True)}); strongest is "
            f"{strongest['name']} ({_fmt(strongest['recall'], percent=True)})."
        )
        if gap > 0.20:
            observations.append(
                f"The {_fmt(gap, percent=True)} class-recall spread is material. "
                "Prioritize the weakest class before presenting the model as uniformly reliable."
            )

    scale_latest = _latest(summary["a1_a3_scale"])
    if scale_latest:
        observations.append(
            f"A1/A3 attention contribution is logged at {scale_latest.value:.2f}; "
            "1.00 means the proximity signals are fully enabled."
        )

    if not observations:
        observations.append(
            "The logs contain too few comparable validation points for trend interpretation."
        )
    return status, observations, class_rows


def _figure_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _plot_lines(
    title: str,
    ylabel: str,
    lines: list[tuple[str, list[Point]]],
) -> Optional[str]:
    lines = [(name, points) for name, points in lines if points]
    if not lines:
        return None
    fig, axis = plt.subplots(figsize=(9.2, 4.4))
    for name, points in lines:
        axis.plot(
            [point.epoch for point in points],
            [point.value for point in points],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=name,
        )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if len(lines) > 1:
        axis.legend(loc="best")
    fig.tight_layout()
    return _figure_data_uri(fig)


def _plot_class_metrics(class_rows: list[dict]) -> Optional[str]:
    available = [
        row
        for row in class_rows
        if _finite(row["recall"]) or _finite(row["iou"])
    ]
    if not available:
        return None
    x = list(range(len(available)))
    width = 0.38
    recalls = [row["recall"] if _finite(row["recall"]) else float("nan") for row in available]
    ious = [row["iou"] if _finite(row["iou"]) else float("nan") for row in available]
    fig, axis = plt.subplots(figsize=(9.2, 4.5))
    axis.bar([value - width / 2 for value in x], recalls, width, label="Recall")
    axis.bar([value + width / 2 for value in x], ious, width, label="IoU")
    axis.set_title("Latest validation performance by class")
    axis.set_xlabel("Face class")
    axis.set_ylabel("Score (0–1)")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x, [row["name"] for row in available], rotation=20)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    return _figure_data_uri(fig)


def _write_metrics_csv(path: Path, series: dict[str, list[Point]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tag", "epoch", "global_step", "value", "wall_time"])
        for tag in sorted(series):
            for point in series[tag]:
                writer.writerow(
                    [tag, f"{point.epoch:.6g}", point.step, f"{point.value:.9g}", point.wall_time]
                )


def _class_table_html(class_rows: list[dict]) -> str:
    rows = []
    for row in class_rows:
        rows.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>{html.escape(row['name'])}</td>"
            f"<td>{_fmt(row['recall'], percent=True)}</td>"
            f"<td>{_fmt(row['iou'], percent=True)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _report_html(
    *,
    run_name: str,
    log_root: Path,
    summary: dict,
    status: str,
    observations: list[str],
    class_rows: list[dict],
    plots: list[tuple[str, Optional[str], str]],
) -> str:
    best = summary["best_selection"]
    latest = summary["latest_selection"]
    latest_macro = _latest(summary["macro"])
    latest_overall = _latest(summary["overall"])
    latest_iou = _latest(summary["iou"])
    latest_eval = _latest(summary["eval_loss"])
    elapsed_hours = max(0.0, summary["end_time"] - summary["start_time"]) / 3600.0
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    best_epoch_text = f"{best.epoch:.0f}" if best else "n/a"
    latest_epoch_text = f"{latest.epoch:.0f}" if latest else "n/a"

    plot_sections = []
    for title, data_uri, caption in plots:
        if data_uri:
            plot_sections.append(
                f"<section><h2>{html.escape(title)}</h2>"
                f"<img src=\"{data_uri}\" alt=\"{html.escape(title)}\">"
                f"<p class=\"caption\">{html.escape(caption)}</p></section>"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training report — {html.escape(run_name)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; color: #17202a; background: #f5f7f9; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 32px 28px 56px; }}
header {{ background: #15283b; color: white; padding: 30px; border-radius: 8px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin-top: 30px; font-size: 20px; }}
.subtitle, .caption {{ color: #5f6b76; font-size: 13px; }}
header .subtitle {{ color: #d5dde5; }}
.status {{ display: inline-block; margin-top: 14px; padding: 7px 11px; background: #e8eef4; color: #15283b; border-radius: 4px; font-weight: 600; }}
.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 20px 0; }}
.kpi {{ background: white; border: 1px solid #dce2e7; padding: 16px; border-radius: 6px; }}
.kpi .label {{ color: #5f6b76; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ font-size: 24px; font-weight: 650; margin-top: 6px; }}
section {{ background: white; border: 1px solid #dce2e7; padding: 20px 22px; margin-top: 16px; border-radius: 6px; }}
img {{ display: block; max-width: 100%; margin: 8px auto; }}
li {{ margin: 8px 0; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ text-align: left; border-bottom: 1px solid #e4e8ec; padding: 10px; }}
th {{ background: #f4f6f8; }}
.note {{ border-left: 4px solid #607d99; padding: 10px 14px; background: #f2f5f8; line-height: 1.45; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body><main>
<header>
  <h1>Model training performance report</h1>
  <div>{html.escape(run_name)}</div>
  <div class="subtitle">Generated {generated} · Source: <code>{html.escape(str(log_root))}</code></div>
  <div class="status">{html.escape(status)}</div>
</header>

<div class="kpis">
  <div class="kpi"><div class="label">Latest macro accuracy</div><div class="value">{_fmt(latest_macro.value if latest_macro else None, True)}</div></div>
  <div class="kpi"><div class="label">Latest overall accuracy</div><div class="value">{_fmt(latest_overall.value if latest_overall else None, True)}</div></div>
  <div class="kpi"><div class="label">Latest mean IoU</div><div class="value">{_fmt(latest_iou.value if latest_iou else None, True)}</div></div>
  <div class="kpi"><div class="label">Latest validation loss</div><div class="value">{_fmt(latest_eval.value if latest_eval else None)}</div></div>
  <div class="kpi"><div class="label">Best {html.escape(summary['selection_name'])}</div><div class="value">{_fmt(best.value if best else None, summary['selection_maximise'])}</div></div>
  <div class="kpi"><div class="label">Best epoch</div><div class="value">{best_epoch_text}</div></div>
  <div class="kpi"><div class="label">Latest logged epoch</div><div class="value">{latest_epoch_text}</div></div>
  <div class="kpi"><div class="label">Logged wall time</div><div class="value">{elapsed_hours:.1f} h</div></div>
</div>

<section>
  <h2>Executive interpretation</h2>
  <ul>{"".join(f"<li>{html.escape(item)}</li>" for item in observations)}</ul>
  <p class="note"><strong>Decision guidance:</strong> validation results guide model selection,
  but they are not final evidence of real-world performance. Present held-out test metrics
  before making a deployment or production-readiness claim.</p>
</section>

{"".join(plot_sections)}

<section>
  <h2>Latest class-level validation metrics</h2>
  <table><thead><tr><th>ID</th><th>Class</th><th>Recall</th><th>IoU</th></tr></thead>
  <tbody>{_class_table_html(class_rows)}</tbody></table>
  <p class="caption">Recall answers “of the true faces in this class, how many were found?”
  IoU is stricter because it penalizes both missed faces and false assignments.</p>
</section>

<section>
  <h2>How to explain the graphs</h2>
  <ul>
    <li><strong>Training loss:</strong> the weighted optimization objective. Falling loss means
    the model is fitting the training data, but it is not a business-facing accuracy measure.</li>
    <li><strong>Validation loss:</strong> the same objective on unseen validation parts. A rising
    validation loss while training loss falls is a warning for overfitting.</li>
    <li><strong>Overall face accuracy:</strong> every face counts equally. Large/frequent classes
    can dominate this number.</li>
    <li><strong>Macro class accuracy:</strong> every class counts equally. This is the most useful
    headline accuracy for the imbalanced stock/thread/text/chamfer/fillet problem.</li>
    <li><strong>Mean IoU:</strong> measures prediction overlap and penalizes false positives and
    false negatives. It is generally harder to score highly than accuracy.</li>
    <li><strong>Per-class metrics:</strong> show whether the model is uniformly useful. Chamfer and
    fillet should be reviewed separately because aggregate metrics can conceal weak rare classes.</li>
  </ul>
</section>
</main></body></html>"""


def _report_markdown(
    run_name: str,
    log_root: Path,
    summary: dict,
    status: str,
    observations: list[str],
    class_rows: list[dict],
) -> str:
    best = summary["best_selection"]
    latest = summary["latest_selection"]
    lines = [
        f"# Model training report — {run_name}",
        "",
        f"- Status: **{status}**",
        f"- TensorBoard source: `{log_root}`",
        f"- Best {summary['selection_name']}: "
        f"**{_fmt(best.value if best else None, summary['selection_maximise'])}** "
        f"at epoch **{best.epoch:.0f}**" if best else
        f"- Best {summary['selection_name']}: not logged",
        f"- Latest logged epoch: **{latest.epoch:.0f}**" if latest else
        "- Latest logged epoch: not available",
        "",
        "## Executive interpretation",
        "",
    ]
    lines.extend(f"- {item}" for item in observations)
    lines.extend(
        [
            "",
            "## Latest class-level validation metrics",
            "",
            "| Class | Recall | IoU |",
            "|---|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {row['name']} | {_fmt(row['recall'], True)} | {_fmt(row['iou'], True)} |"
        )
    lines.extend(
        [
            "",
            "> Validation metrics guide model selection but do not establish final "
            "real-world performance. Use held-out test results for readiness claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-name", help="Stage-1 run name under results/logs/stage1.")
    source.add_argument("--log-dir", type=Path, help="TensorBoard directory or event file.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used with --run-name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Report output directory. Default: results/reports/<run-name>.",
    )
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names in model id order.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    log_root = _resolve_log_root(repo_root, args.run_name, args.log_dir)
    event_dirs = _event_directories(log_root)
    raw = _load_raw_scalars(event_dirs)
    series = _normalise_scalars(raw)
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if not class_names:
        raise SystemExit("--class-names must contain at least one class")

    run_name = args.run_name or log_root.parent.name
    if log_root.name.startswith("version_") and log_root.parent.name == "tensorboard":
        run_name = log_root.parent.parent.name
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (repo_root / "results" / "reports" / run_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _metric_summary(series)
    status, observations, class_rows = _interpret(summary, series, class_names)
    plots = [
        (
            "Training and validation loss",
            _plot_lines(
                "Weighted training and validation loss by epoch",
                "Weighted cross-entropy loss",
                [
                    ("Training loss", summary["train_loss"]),
                    ("Validation loss", summary["eval_loss"]),
                ],
            ),
            "Lower is better. Source: TensorBoard; epoch-level weighted loss.",
        ),
        (
            "Validation quality",
            _plot_lines(
                "Validation segmentation quality by epoch",
                "Score (0–1)",
                [
                    (label, series.get(tag, []))
                    for label, tag in QUALITY_TAGS.items()
                ],
            ),
            "Higher is better. Macro accuracy weights every class equally.",
        ),
        (
            "Class-level performance",
            _plot_class_metrics(class_rows),
            "Latest validation recall and IoU from TensorBoard.",
        ),
        (
            "Learning-rate schedule",
            _plot_lines(
                "Optimizer learning rates by epoch",
                "Learning rate",
                [
                    ("Backbone/classifier LR", summary["current_lr"]),
                    ("A1/A3 LR", summary["a1_a3_lr"]),
                ],
            ),
            "Source: TensorBoard optimizer logs.",
        ),
    ]

    html_path = output_dir / "manager_report.html"
    markdown_path = output_dir / "manager_summary.md"
    csv_path = output_dir / "tensorboard_scalars.csv"
    json_path = output_dir / "summary.json"

    html_path.write_text(
        _report_html(
            run_name=run_name,
            log_root=log_root,
            summary=summary,
            status=status,
            observations=observations,
            class_rows=class_rows,
            plots=plots,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _report_markdown(
            run_name, log_root, summary, status, observations, class_rows
        ),
        encoding="utf-8",
    )
    _write_metrics_csv(csv_path, series)
    json_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "status": status,
                "selection_metric": summary["selection_name"],
                "best_epoch": summary["best_selection"].epoch
                if summary["best_selection"]
                else None,
                "best_value": summary["best_selection"].value
                if summary["best_selection"]
                else None,
                "latest_epoch": summary["latest_selection"].epoch
                if summary["latest_selection"]
                else None,
                "observations": observations,
                "classes": class_rows,
                "tensorboard_tags": sorted(series),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"TensorBoard event directories: {len(event_dirs)}")
    print(f"Scalar tags read:              {len(series)}")
    print(f"Status:                        {status}")
    if summary["best_selection"]:
        print(
            f"Best {summary['selection_name']}: "
            f"{summary['best_selection'].value:.4f} "
            f"(epoch {summary['best_selection'].epoch:.0f})"
        )
    print(f"\nHTML report:     {html_path}")
    print(f"Markdown summary:{markdown_path}")
    print(f"Scalar export:   {csv_path}")
    print(f"JSON summary:    {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
