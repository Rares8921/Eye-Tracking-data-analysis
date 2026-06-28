from pathlib import Path

import numpy as np
import pandas as pd

from classical_models import evaluate_task
from sequence_models import MAX_SEQ_LEN, evaluate_sequence_task
from utils import (
    CLASSICAL_MODELS_OUTPUTS,
    EDA_DIR,
    FEATURES_DIR,
    RECORDINGS_DIR,
    WALDO_DIR,
    add_relative_time_columns,
    long_search_labels,
    participant_id_from_recording,
    recordings as WALDO_INTERVALS,
    saliency_maps as load_saliency_maps,
    saliency_values,
)

OUTPUT_DIR = Path("sensitivity")
WINDOW_S = 5.0

GAZE_ONLY_COLS = [
    "early_fix_count",
    "early_fix_rate",
    "early_avg_fix_duration_s",
    "early_total_fix_duration_s",
    "early_saccade_length_avg",
    "early_saccade_length_median",
    "early_scanpath_length_total",
    "early_scanpath_length_per_s",
    "early_spatial_coverage_hull",
    "early_fixation_entropy_4x4",
    "early_transition_entropy_4x4",
    "early_unique_grid_cells_4x4",
    "early_mean_distance_to_target",
    "early_min_distance_to_target",
    "early_gaze_entropy",
    "early_gaze_count",
    "early_mean_saliency",
    "early_max_saliency",
]

FEATURE_SETS = {
    "gaze_only": GAZE_ONLY_COLS,
    "gaze_plus_target": GAZE_ONLY_COLS
    + [
        "early_waldo_hit_count",
        "early_direct_hit_count",
        "early_peripheral_hit_count",
        "early_hit_any",
    ],
}


def build_quality_exclusion_table() -> pd.DataFrame:
    quality_path = EDA_DIR / "recording_quality_summary.csv"
    if not quality_path.exists():
        return pd.DataFrame()

    quality_df = pd.read_csv(quality_path)
    quality_df["exclude_high_fix_oob"] = quality_df["fix_out_of_bounds_pct"] > 5.0
    quality_df["exclude_high_gaze_oob"] = quality_df["gaze_out_of_bounds_pct"] > 5.0
    quality_df["exclude_truncated_fix"] = quality_df["fix_timestamp_truncated"].fillna(False).astype(bool)
    quality_df["exclude_any_quality"] = (
        quality_df["exclude_high_fix_oob"]
        | quality_df["exclude_high_gaze_oob"]
        | quality_df["exclude_truncated_fix"]
    )
    return quality_df


def add_stream_mismatch_flag(dataset_df: pd.DataFrame) -> pd.DataFrame:
    if dataset_df.empty:
        return dataset_df

    features_path = FEATURES_DIR / "features_summary.csv"
    if not features_path.exists():
        dataset_df["exclude_zero_fix"] = False
        return dataset_df

    features_df = pd.read_csv(features_path)
    if "total_fixations" not in features_df.columns:
        dataset_df["exclude_zero_fix"] = False
        return dataset_df

    merged = dataset_df.merge(
        features_df[["recording_id", "level_name", "total_fixations"]],
        on=["recording_id", "level_name"],
        how="left",
    )
    merged["exclude_zero_fix"] = merged["total_fixations"].fillna(0) <= 0
    return merged


def attach_quality_flags(dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_df = add_stream_mismatch_flag(dataset_df)
    quality_df = build_quality_exclusion_table()

    if not quality_df.empty:
        dataset_df = dataset_df.merge(
            quality_df[["recording_id", "exclude_any_quality"]],
            on="recording_id",
            how="left",
        )
        dataset_df["exclude_any_quality"] = dataset_df["exclude_any_quality"].fillna(False).astype(bool)
    else:
        dataset_df["exclude_any_quality"] = False

    return dataset_df, quality_df


def build_sequence_features(xs, ys, durs, saliency_vals):
    if len(xs) == 0:
        return np.zeros((0, 9), dtype=np.float32)

    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    saccade_len = np.sqrt(dx**2 + dy**2)
    saccade_angle = np.arctan2(dy, dx) / np.pi
    velocity = saccade_len / (durs + 1e-5)

    return np.stack(
        [
            xs,
            ys,
            durs,
            saccade_len,
            saccade_angle,
            velocity,
            np.diff(velocity, prepend=velocity[0]),
            np.diff(saccade_angle, prepend=saccade_angle[0]),
            saliency_vals,
        ],
        axis=1,
    ).astype(np.float32)


def pad_sequence(seq, max_seq_len=MAX_SEQ_LEN):
    seq = seq[:max_seq_len]
    length = len(seq)
    if length < max_seq_len:
        seq = np.vstack([seq, np.zeros((max_seq_len - length, seq.shape[1]), dtype=np.float32)])
    return seq, length


def build_sequence_dataset(window_s=WINDOW_S, max_seq_len=MAX_SEQ_LEN):
    s_maps = load_saliency_maps()
    rows = []
    sequences = []
    lengths = []

    for rec_id, intervals in WALDO_INTERVALS.items():
        fix_path = RECORDINGS_DIR / rec_id / "fixations_on_surface_Surface 1.csv"
        waldo_path = WALDO_DIR / f"waldo_fixations_{rec_id}.csv"
        if not fix_path.exists():
            continue

        fix_df = pd.read_csv(fix_path)
        waldo_df = pd.read_csv(waldo_path) if waldo_path.exists() else pd.DataFrame()
        if fix_df.empty:
            continue

        add_relative_time_columns(
            fix_df,
            start_col="start timestamp [ns]",
            end_col="end timestamp [ns]",
            duration_col="duration [ms]",
        )

        for level_name, start_s, end_s in intervals:
            early_end_s = min(end_s, start_s + window_s)
            level_fix = fix_df[(fix_df["start_s"] >= start_s) & (fix_df["start_s"] <= early_end_s)].copy()
            if len(level_fix) < 3:
                continue

            level_waldo = (
                waldo_df[waldo_df["waldo"] == level_name].copy()
                if not waldo_df.empty and "waldo" in waldo_df.columns
                else pd.DataFrame()
            )

            xs = level_fix["fixation x [normalized]"].to_numpy(dtype=float)
            ys = level_fix["fixation y [normalized]"].to_numpy(dtype=float)
            durs = pd.to_numeric(level_fix["duration [ms]"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0
            seq = build_sequence_features(xs, ys, durs, saliency_values(xs, ys, s_maps.get(level_name)))
            seq_padded, seq_len = pad_sequence(seq, max_seq_len)

            sequences.append(seq_padded)
            lengths.append(seq_len)
            rows.append(
                {
                    "recording_id": rec_id,
                    "participant_id": participant_id_from_recording(rec_id),
                    "level_name": level_name,
                    "window_s": float(window_s),
                    "total_search_time_s": float(end_s - start_s),
                    "label_found_waldo": int(not level_waldo.empty),
                }
            )

    if not rows:
        return pd.DataFrame(), np.zeros((0, max_seq_len, 9), dtype=np.float32), np.zeros((0,), dtype=int)

    meta_df = long_search_labels(pd.DataFrame(rows))
    return meta_df, np.asarray(sequences, dtype=np.float32), np.asarray(lengths, dtype=np.int64)


def scenario_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "all_data": np.ones(len(df), dtype=bool),
        "drop_zero_fix_levels": ~df["exclude_zero_fix"].to_numpy(),
        "drop_quality_flagged_participants": ~df["exclude_any_quality"].to_numpy(),
        "drop_both": ((~df["exclude_zero_fix"]) & (~df["exclude_any_quality"])).to_numpy(),
    }


def evaluate_tabular_scenarios() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = CLASSICAL_MODELS_OUTPUTS / f"early_window_dataset_{int(WINDOW_S)}s.csv"
    if not dataset_path.exists():
        return pd.DataFrame(), build_quality_exclusion_table()

    dataset_df, quality_df = attach_quality_flags(pd.read_csv(dataset_path))
    rows = []

    for scenario_name, mask in scenario_masks(dataset_df).items():
        scenario_df = dataset_df.loc[mask].copy()
        if scenario_df.empty:
            continue
        groups = scenario_df["participant_id"].astype(str)

        for task, feature_key in [
            ("label_found_waldo", "gaze_plus_target"),
            ("label_long_search", "gaze_only"),
        ]:
            result_df, _ = evaluate_task(scenario_df, FEATURE_SETS[feature_key], task, groups)
            if result_df.empty:
                continue
            best = result_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
            rows.append(
                {
                    "scenario": scenario_name,
                    "family": "tabular",
                    "task": task,
                    "best_model": best["model"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "n_samples": int(best["n_samples"]),
                }
            )

    return pd.DataFrame(rows), quality_df


def evaluate_sequence_scenarios() -> pd.DataFrame:
    meta_df, sequences, lengths = build_sequence_dataset()
    if meta_df.empty:
        return pd.DataFrame()

    meta_df, _ = attach_quality_flags(meta_df)
    rows = []

    for scenario_name, mask in scenario_masks(meta_df).items():
        scenario_meta = meta_df.loc[mask].reset_index(drop=True)
        if scenario_meta.empty:
            continue
        scenario_seq = sequences[mask]
        scenario_lengths = lengths[mask]

        for task in ["label_found_waldo", "label_long_search"]:
            result_df, _ = evaluate_sequence_task(scenario_meta, scenario_seq, scenario_lengths, task)
            if result_df.empty:
                continue
            best = result_df.sort_values("balanced_accuracy", ascending=False).iloc[0]
            rows.append(
                {
                    "scenario": scenario_name,
                    "family": "sequence",
                    "task": task,
                    "best_model": best["model"],
                    "balanced_accuracy": best["balanced_accuracy"],
                    "n_samples": int(best["n_samples"]),
                }
            )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    tabular_df, quality_df = evaluate_tabular_scenarios()
    sequence_df = evaluate_sequence_scenarios()

    quality_df.to_csv(OUTPUT_DIR / "quality_exclusion_flags.csv", index=False)
    tabular_df.to_csv(OUTPUT_DIR / "tabular_sensitivity.csv", index=False)
    sequence_df.to_csv(OUTPUT_DIR / "sequence_sensitivity.csv", index=False)
