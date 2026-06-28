import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import (
    EDA_DIR,
    EXPLAINABILITY_DIR,
    EXTRA_FIGURES_DIR as OUTPUT_DIR,
    RECORDINGS_DIR,
    SENSITIVITY_DIR,
    recordings as WALDO_INTERVALS,
    participant_id_from_recording,
    add_relative_time_columns
)


task_labels = {
    "label_found_waldo": "Waldo found",
    "label_long_search": "Long search",
}

scenario_labels = {
    "all_data": "all data",
    "drop_zero_fix": "no zero-fix",
    "drop_zero_fix_levels": "no zero-fix",
    "drop_quality": "no quality flags",
    "drop_quality_flagged_participants": "no quality flags",
    "drop_both": "drop both",
}

ablation_labels = {
    "target_contact": "target contact",
    "entropy_structure": "entropy",
    "spatial_search": "spatial search",
    "gaze_dynamics": "gaze dynamics",
    "fixation_dynamics": "fixation dynamics",
    "saliency_alignment": "saliency",
    "quality_context": "quality",
}

participant_labels = {
    participant_id_from_recording(rec_id): f"Participant {idx + 1}"
    for idx, rec_id in enumerate(WALDO_INTERVALS)
}


def save_figure(path, **kwargs):
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(path, **kwargs)
    plt.savefig(path.with_suffix(".pdf"), **{k: v for k, v in kwargs.items() if k != "dpi"})


def make_ablation_plot():
    path = EXPLAINABILITY_DIR / "feature_group_ablation.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if df.empty or "ablation" not in df.columns:
        return None

    df = df[df["ablation"] != "none"].copy()
    if df.empty:
        return None

    df["task"] = df["task"].map(task_labels).fillna(df["task"])
    df["ablation"] = df["ablation"].map(ablation_labels).fillna(df["ablation"])

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="ablation", y="score_drop", hue="task")
    plt.xlabel("removed group")
    plt.ylabel("balanced accuracy drop")
    plt.title("Feature group ablation")
    plt.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "ablation_score_drop.png"
    save_figure(out_path, dpi=220)
    plt.close()
    return out_path


def read_sensitivity_tables():
    tab_paths = [
        SENSITIVITY_DIR / "tabular_sensitivity.csv",
        SENSITIVITY_DIR / "classical_sensitivity.csv",
    ]
    seq_paths = [
        SENSITIVITY_DIR / "sequence_sensitivity.csv",
    ]

    frames = []
    for path in tab_paths:
        if path.exists():
            df = pd.read_csv(path)
            df["family"] = "tabular"
            frames.append(df)
            break

    for path in seq_paths:
        if path.exists():
            df = pd.read_csv(path)
            df["family"] = "sequence"
            frames.append(df)
            break

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_sensitivity_plot():
    df = read_sensitivity_tables()
    if df.empty:
        return None

    df["task"] = df["task"].map(task_labels).fillna(df["task"])
    df["scenario"] = df["scenario"].map(scenario_labels).fillna(df["scenario"])
    df["label"] = df["family"] + " / " + df["task"]

    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=df, x="scenario", y="balanced_accuracy", hue="label")
    plt.xlabel("filtering scenario")
    plt.ylabel("balanced accuracy")
    plt.title("Sensitivity analysis")
    plt.ylim(0, 1)
    plt.legend(title="")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "sensitivity_balacc.png"
    save_figure(out_path, dpi=220)
    plt.close()
    return out_path


def quadrant_id(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    return (ys >= 0.5).astype(int) * 2 + (xs >= 0.5).astype(int)


def aggregate_transition_matrix(level_name):
    counts = np.zeros((4, 4), dtype=float)

    for rec_id, intervals in WALDO_INTERVALS.items():
        interval = [item for item in intervals if item[0] == level_name]
        if not interval:
            continue

        fix_path = RECORDINGS_DIR / rec_id / "fixations_on_surface_Surface 1.csv"
        if not fix_path.exists():
            continue

        df = pd.read_csv(fix_path)
        if df.empty:
            continue

        add_relative_time_columns(
            df,
            start_col="start timestamp [ns]",
            end_col="end timestamp [ns]",
            duration_col="duration [ms]",
        )

        _, start_s, end_s = interval[0]
        sub = df[(df["start_s"] >= start_s) & (df["start_s"] <= end_s)].copy()
        if len(sub) < 2:
            continue

        quads = quadrant_id(sub["fixation x [normalized]"], sub["fixation y [normalized]"])
        for src, dst in zip(quads[:-1], quads[1:]):
            counts[int(src), int(dst)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)


def make_transition_heatmaps():
    paths = []
    labels = ["UL", "UR", "LL", "LR"]

    for level_name in ["waldo_1", "waldo_2", "waldo_3", "waldo_4"]:
        mat = aggregate_transition_matrix(level_name)
        if mat.sum() <= 0:
            continue

        plt.figure(figsize=(5.5, 4.5))
        sns.heatmap(mat.T, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=max(0.25, mat.max()))
        plt.xticks(np.arange(4) + 0.5, labels)
        plt.yticks(np.arange(4) + 0.5, labels, rotation=0)
        plt.xlabel("from quadrant")
        plt.ylabel("to quadrant")
        plt.title(f"Transitions: {level_name}")
        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{level_name}_transition_matrix.png"
        save_figure(out_path, dpi=220)
        plt.close()
        paths.append(out_path)

    return paths


def make_consistency_plot():
    path = EDA_DIR / "participant_consistency_summary.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    if "annotated_duration_s_cv" not in df.columns:
        return None

    df = df.sort_values("annotated_duration_s_cv")
    df["participant_label"] = df["participant_id"].map(participant_labels).fillna(df["participant_id"])

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="participant_label", y="annotated_duration_s_cv", color="#4C78A8")
    plt.xlabel("participant")
    plt.ylabel("duration coefficient of variation")
    plt.title("Participant consistency")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "participant_consistency_cv.png"
    save_figure(out_path, dpi=220)
    plt.close()
    return out_path


def write_summary(paths):
    lines = ["# Extra figures summary", ""]
    for path in paths:
        lines.append(f"- {path.name}")
    (OUTPUT_DIR / "extra_figures_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    made = []
    for path in [make_ablation_plot(), make_sensitivity_plot(), make_consistency_plot()]:
        if path is not None:
            made.append(path)
    made.extend(make_transition_heatmaps())

    write_summary(made)
    print(f": {len(made)}")
