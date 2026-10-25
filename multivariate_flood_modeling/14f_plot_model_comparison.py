#!/usr/bin/env python3
"""14f — Plot saved classification and incremental-driver results.

No training or threshold selection is performed. Dependencies: Python >=3.9,
matplotlib >=3.5, numpy >=1.21.

Place this file in scripts/multivariate_flood_modeling/ and run:
    python scripts/multivariate_flood_modeling/14f_plot_model_comparison.py

Optional explicit project root:
    python 14f_plot_model_comparison.py --root /path/to/DBL-Group-22

--root also accepts the multivariate_flood_modeling results folder itself.
Inputs:
    processed_data/multivariate_flood_modeling/14b_monthly_prediction_test/
        model_test_summary.json
    14c_weekly_prediction_test/model_test_summary.json
    14d_same_month_next_year_prediction_test/model_test_summary.json
    14e_same_week_next_year_prediction_test/model_test_summary.json
    10b_monthly_rolling_origin/04_pooled_test_metrics.csv
Default output:
    processed_data/multivariate_flood_modeling/14f_model_comparison_figures/
        01_model_comparison.png
        01_model_comparison.pdf
        02_values_used.csv
        03_figure_metadata.json

Panel a: compare F1 WITHIN each forecast task. Different horizons have different
labels/cohorts and are not a controlled experiment on lead time alone.
Panel b: ADM3, C_combined, Monthly Rolling; M0_history vs M4_all_drivers.
The target is a recorded-event-date proxy, not validated inundation duration.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TASKS = [
    ("Next month", "14b_monthly_prediction_test"),
    ("Next week", "14c_weekly_prediction_test"),
    ("Same month\nnext year", "14d_same_month_next_year_prediction_test"),
    ("Same week\nnext year", "14e_same_week_next_year_prediction_test"),
]
MODELS = [
    ("persistence", "Persistence", "#8B929A"),
    ("logistic_regression", "Logistic regression", "#007C91"),
    ("random_forest", "Random Forest", "#B65439"),
]
DRIVER_CSV = Path("10b_monthly_rolling_origin/04_pooled_test_metrics.csv")


def candidate_data_dirs(root):
    return [
        root / "processed_data" / "multivariate_flood_modeling",
        root / "multivariate_flood_modeling",
        root,
    ]


def resolve_data_dir(root):
    """Find a complete input bundle; never silently switch explicit roots."""
    if root is not None:
        roots = [root.expanduser().resolve()]
    else:
        here = Path(__file__).resolve().parent
        roots = list(dict.fromkeys([Path.cwd(), *Path.cwd().parents,
                                   here, *here.parents]))
    checked = []
    for base in roots:
        for candidate in candidate_data_dirs(base):
            if candidate in checked:
                continue
            checked.append(candidate)
            required = [candidate / folder / "model_test_summary.json"
                        for _, folder in TASKS]
            required.append(candidate / DRIVER_CSV)
            if all(path.is_file() for path in required):
                return candidate.resolve()
    message = (
        "Cannot find all 14b–14e summaries and the 10b pooled metrics. "
        "Run the earlier analyses first, or pass --root /path/to/DBL-Group-22 "
        "(or --root /path/to/processed_data/multivariate_flood_modeling)."
    )
    if root is not None:
        for candidate in candidate_data_dirs(roots[0]):
            if candidate.is_dir():
                required = [Path(folder) / "model_test_summary.json"
                            for _, folder in TASKS] + [DRIVER_CSV]
                missing = [str(p) for p in required
                           if not (candidate / p).is_file()]
                message += "\nChecked: " + str(candidate)
                message += "\nMissing: " + ", ".join(missing)
    raise FileNotFoundError(message)


def finite_number(value, description, lower=None, upper=None):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} is not finite.")
    if lower is not None and result < lower:
        raise ValueError(f"{description} is below {lower}.")
    if upper is not None and result > upper:
        raise ValueError(f"{description} exceeds {upper}.")
    return result


def read_values(data_dir):
    """Return chart data with their source paths and cohort sizes."""
    rows, warnings = [], []
    for task, folder in TASKS:
        source = data_dir / folder / "model_test_summary.json"
        with source.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        task_counts = set()
        for key, label, _ in MODELS:
            try:
                metrics = summary[key]["test_metrics"]
                f1 = finite_number(metrics["f1"], f"{folder}/{key}/F1", 0, 1)
                n = int(metrics["n"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid model metrics in {source}: {key}") from error
            if n <= 0:
                raise ValueError(f"Non-positive sample size in {source}: {key}")
            task_counts.add(n)
            rows.append({
                "panel": "a", "task": task.replace("\n", " "),
                "model": key, "metric": "F1", "value": f1, "n": n,
                "source": str(source.relative_to(data_dir)),
            })
            if key == "persistence" and int(metrics.get("TN", -1)) == 0:
                negatives = int(metrics.get("negatives", -1))
                if negatives > 0 and int(metrics.get("FP", -2)) == negatives:
                    warnings.append(
                        f"{folder}: persistence labels every negative positive "
                        "at its stored threshold; review the baseline design."
                    )
        if len(task_counts) != 1:
            raise ValueError(f"Model sample sizes differ within task: {folder}")

    source = data_dir / DRIVER_CSV
    with source.open(newline="", encoding="utf-8-sig") as handle:
        pooled = list(csv.DictReader(handle))
    driver_counts = set()
    for key in ["M0_history", "M4_all_drivers"]:
        matches = [
            row for row in pooled
            if row.get("level") == "ADM3"
            and row.get("memory") == "C_combined"
            and row.get("strategy") == "Monthly Rolling"
            and row.get("model") == key
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one ADM3/C_combined/Monthly Rolling/{key} "
                f"row in {source}; found {len(matches)}."
            )
        record = matches[0]
        mae = finite_number(record["MAE"], f"{key}/MAE", lower=0)
        n = int(record["test_n"])
        if n <= 0:
            raise ValueError(f"Invalid driver sample size for {key}.")
        driver_counts.add(n)
        rows.append({
            "panel": "b", "task": "Combined memory; monthly updating",
            "model": key, "metric": "MAE", "value": mae, "n": n,
            "source": str(DRIVER_CSV),
        })
    if len(driver_counts) != 1:
        raise ValueError("Driver models do not share the same sample size.")
    # Equal counts alone do not prove identical cases. The 10b methodology
    # defines the common complete-case cohort; this script does not reconstruct it.
    return rows, warnings


def draw_figure(rows, output_dir, dpi):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(
        1, 2, figsize=(12.6, 4.7),
        gridspec_kw={"width_ratios": [1.65, 1]},
    )
    positions = np.arange(len(TASKS))
    for index, (key, label, colour) in enumerate(MODELS):
        values = [
            next(row["value"] for row in rows
                 if row["panel"] == "a"
                 and row["task"] == task.replace("\n", " ")
                 and row["model"] == key)
            for task, _ in TASKS
        ]
        bars = axes[0].bar(
            positions + (index - 1) * .24, values, width=.23,
            color=colour, label=label,
        )
        for bar, value in zip(bars, values):
            axes[0].text(
                bar.get_x() + bar.get_width()/2, value + .012,
                f"{value:.3f}", ha="center", va="bottom", fontsize=8,
            )
    axes[0].set_xticks(positions, [task for task, _ in TASKS])
    axes[0].set_ylim(0, 1.01)
    axes[0].set_ylabel("F1 score")
    axes[0].set_title(
        "a  Classification across forecast horizons", loc="left", pad=18,
        fontsize=12,
    )
    axes[0].legend(
        loc="upper center", bbox_to_anchor=(.52, -.2),
        ncol=3, frameon=False, fontsize=9,
    )

    values = [next(row["value"] for row in rows
                   if row["panel"] == "b" and row["model"] == key)
              for key in ["M0_history", "M4_all_drivers"]]
    bars = axes[1].bar([0, 1], values, color=["#007C91", "#8B929A"], width=.58)
    ceiling = max(values) * 1.23 if max(values) > 0 else 1
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width()/2, value + ceiling*.015,
            f"{value:.3f}", ha="center", va="bottom", fontsize=11,
        )
    axes[1].set_xticks(
        [0, 1], ["History only", "History + rainfall,\nrunoff and ET0"],
    )
    axes[1].set_ylim(0, ceiling)
    axes[1].set_ylabel("MAE in recorded event dates")
    axes[1].set_title(
        "b  Incremental driver comparison", loc="left", pad=18, fontsize=12,
    )
    figure.subplots_adjust(
        left=.065, right=.985, top=.85, bottom=.26, wspace=.38,
    )
    for extension in ["png", "pdf"]:
        figure.savefig(
            output_dir / f"01_model_comparison.{extension}",
            dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=.18,
        )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, help="Project root or results directory")
    parser.add_argument("--outdir", type=Path, help="Override the output directory")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution; default 300")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    try:
        data_dir = resolve_data_dir(args.root)
        rows, warnings = read_values(data_dir)
        output_dir = (args.outdir.expanduser().resolve() if args.outdir
                      else data_dir / "14f_model_comparison_figures")
        output_dir.mkdir(parents=True, exist_ok=True)
        draw_figure(rows, output_dir, args.dpi)
        with (output_dir / "02_values_used.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "script": Path(__file__).name,
            "input_directory": str(data_dir),
            "source_metrics_modified": False,
            "models_retrained": False,
            "metric_rounding": "3 decimals for display; full precision in CSV",
            "warnings": warnings,
            "interpretation": [
                "F1: higher is better; MAE: lower is better.",
                "Compare classifiers within each task; tasks have different targets/cohorts.",
                "Panel b filters ADM3, C_combined, Monthly Rolling exactly.",
                "10b specifies a common cohort; this plot checks counts, not case identity.",
                "The flood outcome is a recorded-event-date proxy, not verified flood duration.",
                "River geometry and exposure factors are not classifier inputs here.",
            ],
        }
        with (output_dir / "03_figure_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        print("Inputs:", data_dir)
        print("Saved:", output_dir / "01_model_comparison.png")
        print("Saved:", output_dir / "01_model_comparison.pdf")
        print("Saved: values CSV and figure metadata")
        for warning in warnings:
            print("Source result note:", warning)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"14f failed: {error}\n")


if __name__ == "__main__":
    main()
