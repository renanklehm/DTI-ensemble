from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_COLUMNS = {
    "GSDTI": "gsdti_label",
    "DCGAT-DTI": "dcgat_label",
    "Ensemble": "ensemble_label",
}
CONFUSION_LABELS = ("TP", "FP", "FN", "TN")
REGION_MODEL_MEMBERSHIP = {
    "GSDTI_only": ("GSDTI",),
    "DCGAT-DTI_only": ("DCGAT-DTI",),
    "Ensemble_only": ("Ensemble",),
    "GSDTI_and_DCGAT-DTI_only": ("GSDTI", "DCGAT-DTI"),
    "GSDTI_and_Ensemble_only": ("GSDTI", "Ensemble"),
    "DCGAT-DTI_and_Ensemble_only": ("DCGAT-DTI", "Ensemble"),
    "All_three": ("GSDTI", "DCGAT-DTI", "Ensemble"),
}


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if len(frame.columns) == 1:
            frame = pd.read_csv(path, sep="|")
        return frame
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported table format: {path}")


def latest_prediction_file(root: Path) -> Path:
    candidates = sorted(root.glob("*/test_predictions.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No test_predictions.parquet files found under {root}")
    return max(candidates, key=lambda path: (path.parent.name, path.stat().st_mtime))


def table_info(path: Path) -> dict[str, object]:
    frame = read_table(path)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }


def validate_predictions(frame: pd.DataFrame) -> None:
    required = {"target", *MODEL_COLUMNS.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    for column in required:
        values = set(pd.Series(frame[column]).dropna().astype(int).unique())
        if values - {0, 1}:
            raise ValueError(f"{column} must contain only binary values; got {sorted(values)}")


def bucket_for(target: int, prediction: int) -> str:
    if target == 1 and prediction == 1:
        return "TP"
    if target == 0 and prediction == 1:
        return "FP"
    if target == 0 and prediction == 0:
        return "TN"
    return "FN"


def confusion_sets(frame: pd.DataFrame) -> dict[str, dict[str, set[int]]]:
    targets = frame["target"].astype(int)
    sets_by_model: dict[str, dict[str, set[int]]] = {
        model: {label: set() for label in CONFUSION_LABELS}
        for model in MODEL_COLUMNS
    }
    for model, column in MODEL_COLUMNS.items():
        predictions = frame[column].astype(int)
        for index, (target, prediction) in enumerate(zip(targets, predictions, strict=True)):
            sets_by_model[model][bucket_for(int(target), int(prediction))].add(index)
    return sets_by_model


def region_counts(sets_by_model: dict[str, dict[str, set[int]]]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for label in CONFUSION_LABELS:
        gsdti = sets_by_model["GSDTI"][label]
        dcgat = sets_by_model["DCGAT-DTI"][label]
        ensemble = sets_by_model["Ensemble"][label]
        rows[label] = {
            "GSDTI_total": len(gsdti),
            "DCGAT-DTI_total": len(dcgat),
            "Ensemble_total": len(ensemble),
            "GSDTI_only": len(gsdti - dcgat - ensemble),
            "DCGAT-DTI_only": len(dcgat - gsdti - ensemble),
            "Ensemble_only": len(ensemble - gsdti - dcgat),
            "GSDTI_and_DCGAT-DTI_only": len((gsdti & dcgat) - ensemble),
            "GSDTI_and_Ensemble_only": len((gsdti & ensemble) - dcgat),
            "DCGAT-DTI_and_Ensemble_only": len((dcgat & ensemble) - gsdti),
            "All_three": len(gsdti & dcgat & ensemble),
        }
    return rows


def subsets_for_venn(counts: dict[str, int]) -> tuple[int, int, int, int, int, int, int]:
    return (
        counts["GSDTI_only"],
        counts["DCGAT-DTI_only"],
        counts["GSDTI_and_DCGAT-DTI_only"],
        counts["Ensemble_only"],
        counts["GSDTI_and_Ensemble_only"],
        counts["DCGAT-DTI_and_Ensemble_only"],
        counts["All_three"],
    )


def load_target_totals(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    targets: dict[str, dict[str, float]] = {}
    for model, values in raw.items():
        for label, value in values.items():
            targets.setdefault(label, {})[model] = float(value)
    expected_models = set(MODEL_COLUMNS)
    for label in CONFUSION_LABELS:
        missing = expected_models - set(targets.get(label, {}))
        if missing:
            raise ValueError(f"Missing target totals for {label}: {sorted(missing)}")
    return targets


def scale_counts_to_targets(
    counts: dict[str, dict[str, int]],
    targets: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    scaled: dict[str, dict[str, float]] = {}
    for label in CONFUSION_LABELS:
        region_values = {
            region: max(float(counts[label][region]), 1.0e-9)
            for region in REGION_MODEL_MEMBERSHIP
        }
        for _ in range(500):
            max_delta = 0.0
            for model in MODEL_COLUMNS:
                current = sum(
                    value
                    for region, value in region_values.items()
                    if model in REGION_MODEL_MEMBERSHIP[region]
                )
                target = targets[label][model]
                if current == 0.0:
                    continue
                factor = target / current
                for region in REGION_MODEL_MEMBERSHIP:
                    if model in REGION_MODEL_MEMBERSHIP[region]:
                        region_values[region] *= factor
                max_delta = max(max_delta, abs(current - target))
            if max_delta < 1.0e-6:
                break
        scaled[label] = {
            "GSDTI_total": targets[label]["GSDTI"],
            "DCGAT-DTI_total": targets[label]["DCGAT-DTI"],
            "Ensemble_total": targets[label]["Ensemble"],
            **region_values,
        }
    return scaled


def display_count(value: float) -> str:
    return f"{int(round(value)):,}"


def draw_venn(ax, label: str, counts: dict[str, float]) -> None:
    from matplotlib_venn import venn3

    layout_algorithm = None
    try:
        from matplotlib_venn.layout.venn3 import cost_based

        layout_algorithm = cost_based.LayoutAlgorithm()
    except ModuleNotFoundError:
        pass

    labels = (
        f"GSDTI\nn={display_count(counts['GSDTI_total'])}",
        f"DCGAT-DTI\nn={display_count(counts['DCGAT-DTI_total'])}",
        f"Ensemble\nn={display_count(counts['Ensemble_total'])}",
    )
    venn = venn3(
        subsets=subsets_for_venn(counts),
        set_labels=labels,
        ax=ax,
        layout_algorithm=layout_algorithm,
        subset_label_formatter=display_count,
    )
    colors = {
        "100": "#3b82f6",
        "010": "#f97316",
        "001": "#16a34a",
        "110": "#a855f7",
        "101": "#0891b2",
        "011": "#84cc16",
        "111": "#111827",
    }
    for region_id, color in colors.items():
        patch = venn.get_patch_by_id(region_id)
        if patch is not None:
            patch.set_color(color)
            patch.set_alpha(0.46)
            patch.set_edgecolor("#1f2937")
            patch.set_linewidth(1.1)
    region_label_positions = {
        "100": (-0.62, -0.02),
        "010": (0.62, -0.02),
        "001": (0.0, -0.48),
        "110": (0.0, 0.46),
        "101": (-0.32, -0.28),
        "011": (0.32, -0.28),
        "111": (0.0, 0.0),
    }
    for region_id, position in region_label_positions.items():
        text = venn.get_label_by_id(region_id)
        if text is not None:
            text.set_position(position)
            text.set_ha("center")
            text.set_va("center")
            text.set_fontsize(10)
    set_label_positions = ((-0.72, 0.58), (0.72, 0.58), (0.0, -0.72))
    for text, position in zip(venn.set_labels, set_label_positions, strict=True):
        if text is not None:
            text.set_position(position)
            text.set_ha("center")
            text.set_va("center")
            text.set_fontsize(12)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    ax.set_xlim(min(x_min, -0.9), max(x_max, 0.9))
    ax.set_ylim(min(y_min, -0.85), max(y_max, 0.75))
    ax.set_title(label, fontsize=18, fontweight="bold")


def write_outputs(
    output_dir: Path,
    prediction_path: Path,
    frame: pd.DataFrame,
    counts: dict[str, dict[str, float]],
    gsdti_info: dict[str, object],
    dcgat_info: dict[str, object],
    universe_label: str,
    target_totals: dict[str, dict[str, float]] | None,
) -> None:
    import matplotlib.pyplot as plt

    displayed_universe_count = len(frame)
    if target_totals is not None:
        model_totals = [
            sum(target_totals[label][model] for label in CONFUSION_LABELS)
            for model in MODEL_COLUMNS
        ]
        if max(model_totals) - min(model_totals) < 1.0e-6:
            displayed_universe_count = int(round(model_totals[0]))

    output_dir.mkdir(parents=True, exist_ok=True)
    counts_frame = pd.DataFrame.from_dict(counts, orient="index")
    counts_frame.index.name = "confusion_label"
    counts_frame.to_csv(output_dir / "confusion_overlap_counts.csv")

    metadata = {
        "prediction_file": str(prediction_path),
        "prediction_rows": int(len(frame)),
        "displayed_universe_count": displayed_universe_count,
        "base_inputs": {
            "gsdti": gsdti_info,
            "dcgat_dti": dcgat_info,
        },
        "row_identity": "zero-based row number in the prediction file",
        "universe_label": universe_label,
        "model_label_columns": MODEL_COLUMNS,
        "target_totals": target_totals,
        "counts": counts,
    }
    (output_dir / "confusion_overlap_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    for label in CONFUSION_LABELS:
        fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
        draw_venn(ax, label, counts[label])
        fig.text(
            0.5,
            0.02,
            f"Universe: {displayed_universe_count:,} {universe_label}",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(output_dir / f"{label.lower()}_venn.png")
        fig.savefig(output_dir / f"{label.lower()}_venn.svg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=160)
    for ax, label in zip(axes.ravel(), CONFUSION_LABELS, strict=True):
        draw_venn(ax, label, counts[label])
    fig.suptitle(
        f"Confusion-Label Overlap by Model ({universe_label})",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        f"Universe: {displayed_universe_count:,} {universe_label}",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_dir / "confusion_venn_grid.png")
    fig.savefig(output_dir / "confusion_venn_grid.svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TP/FP/TN/FN Venn diagrams for GSDTI, DCGAT-DTI, and Ensemble predictions."
    )
    parser.add_argument(
        "--ensemble-root",
        type=Path,
        default=Path("artifacts/ensemble"),
        help="Directory containing timestamped Ensemble run folders.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Specific test_predictions.parquet file. Defaults to the latest run under --ensemble-root.",
    )
    parser.add_argument(
        "--gsdti-input",
        type=Path,
        default=Path("data/0_raw/gsdti.parquet"),
        help="GSDTI source prediction table used for provenance.",
    )
    parser.add_argument(
        "--dcgat-input",
        type=Path,
        default=Path("data/0_raw/dcgat-dti.csv"),
        help="DCGAT-DTI source prediction table used for provenance.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to artifacts/confusion_venns/<ensemble-run>.",
    )
    parser.add_argument(
        "--universe-label",
        default="latest Ensemble test predictions",
        help="Human-readable description of the scored row universe.",
    )
    parser.add_argument(
        "--target-totals-json",
        type=Path,
        help="Optional model-level confusion totals. Region overlaps are proportionally rescaled to match them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_path = args.predictions or latest_prediction_file(args.ensemble_root)
    prediction_path = prediction_path.resolve()
    output_dir = args.output_dir or Path("artifacts/confusion_venns") / prediction_path.parent.name

    frame = read_table(prediction_path)
    validate_predictions(frame)
    sets_by_model = confusion_sets(frame)
    counts = region_counts(sets_by_model)
    target_totals = load_target_totals(args.target_totals_json) if args.target_totals_json else None
    if target_totals is not None:
        counts = scale_counts_to_targets(counts, target_totals)

    write_outputs(
        output_dir=output_dir,
        prediction_path=prediction_path,
        frame=frame,
        counts=counts,
        gsdti_info=table_info(args.gsdti_input),
        dcgat_info=table_info(args.dcgat_input),
        universe_label=args.universe_label,
        target_totals=target_totals,
    )
    print(f"Wrote confusion Venn outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
