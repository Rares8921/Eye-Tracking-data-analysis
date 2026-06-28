import os
import pandas as pd
import numpy as np

from utils import (
    FEATURES_DIR as OUTPUT_DIR,
    RECORDINGS_DIR,
    waldo_coords as WALDO_BOXES,
    WALDO_DIR,
    recordings as WALDO_INTERVALS,
    add_relative_time_columns,
    parse_neon_summary_csv,
    cell_entropy,
    convex_hull_area,
    gaze_entropy,
    grid_cell_ids,
    transition_entropy,
)


os.makedirs(OUTPUT_DIR, exist_ok=True)
summary = []

if not os.path.exists(RECORDINGS_DIR):
    print(f"Directory {RECORDINGS_DIR} not found.")
else:
    for rec_id in os.listdir(RECORDINGS_DIR):
        rec_path = os.path.join(RECORDINGS_DIR, rec_id)
        if not os.path.isdir(rec_path):
            continue

        fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
        gaze_file = os.path.join(rec_path, "gaze_positions_on_surface_Surface 1.csv")
        waldo_file = os.path.join(WALDO_DIR, f"waldo_fixations_{rec_id}.csv")

        if not (os.path.exists(fix_file) and os.path.exists(gaze_file) and os.path.exists(waldo_file)):
            continue

        full_fix_df = pd.read_csv(fix_file)
        full_gaze_df = pd.read_csv(gaze_file)
        full_waldo_df = pd.read_csv(waldo_file)

        if not full_fix_df.empty:
            add_relative_time_columns(
                full_fix_df,
                start_col="start timestamp [ns]",
                end_col="end timestamp [ns]",
                duration_col="duration [ms]",
            )
        if not full_gaze_df.empty and 'timestamp [ns]' in full_gaze_df.columns:
            add_relative_time_columns(full_gaze_df, start_col="timestamp [ns]")

        vis_frames, total_frames = parse_neon_summary_csv(os.path.join(rec_path, "surface_visibility.csv"))
        surf_gaze, total_gaze = parse_neon_summary_csv(os.path.join(rec_path, "surface_gaze_distribution.csv"))
        global_vis_ratio = vis_frames / total_frames if total_frames > 0 else 0.0
        global_gaze_ratio = surf_gaze / total_gaze if total_gaze > 0 else 0.0

        rec_out_dir = os.path.join(OUTPUT_DIR, rec_id)
        os.makedirs(rec_out_dir, exist_ok=True)

        intervale = WALDO_INTERVALS.get(rec_id, [])

        for aoi_name, start_s, end_s in intervale:
            level_duration_s = end_s - start_s

            mask_fix = (full_fix_df["start_s"] >= start_s) & (full_fix_df["start_s"] <= end_s)
            fix_df = full_fix_df[mask_fix].copy()

            if not full_gaze_df.empty and "start_s" in full_gaze_df.columns:
                mask_gaze = (full_gaze_df["start_s"] >= start_s) & (full_gaze_df["start_s"] <= end_s)
                gaze_df = full_gaze_df[mask_gaze].copy()
            else:
                gaze_df = pd.DataFrame()

            if not full_waldo_df.empty and "waldo" in full_waldo_df.columns:
                waldo_df = full_waldo_df[full_waldo_df["waldo"] == aoi_name].copy()
            else:
                waldo_df = pd.DataFrame()

            total_fix = len(fix_df)

            if total_fix > 0:
                fix_durations = pd.to_numeric(fix_df["duration [ms]"], errors="coerce").fillna(0.0) / 1000.0
                total_duration = fix_durations.sum()
                avg_fix_dur = fix_durations.mean()

                fix_x = fix_df["fixation x [normalized]"]
                fix_y = fix_df["fixation y [normalized]"]
                fixation_dispersion = fix_df[["fixation x [normalized]", "fixation y [normalized]"]].std().mean()
                revisits = fix_df[["fixation x [normalized]", "fixation y [normalized]"]].round(2).duplicated().sum()
                saccade_lengths = np.sqrt(np.diff(fix_x) ** 2 + np.diff(fix_y) ** 2) if total_fix > 1 else np.array([])
                scanpath_length_total = float(saccade_lengths.sum()) if len(saccade_lengths) else 0.0
                saccade_len = float(saccade_lengths.mean()) if len(saccade_lengths) else 0.0
                saccade_len_median = float(np.median(saccade_lengths)) if len(saccade_lengths) else 0.0
                spatial_coverage_hull = convex_hull_area(fix_x.to_numpy(dtype=float), fix_y.to_numpy(dtype=float))
                grid_ids = grid_cell_ids(fix_x.to_numpy(dtype=float), fix_y.to_numpy(dtype=float), grid_size=4)
                fixation_grid_entropy = cell_entropy(grid_ids, grid_size=4)
                transition_entropy = transition_entropy(grid_ids, grid_size=4)
                unique_grid_cells = int(len(np.unique(grid_ids)))
            else:
                total_duration = avg_fix_dur = fixation_dispersion = revisits = 0.0
                fix_x, fix_y = pd.Series(dtype=float), pd.Series(dtype=float)
                scanpath_length_total = saccade_len = saccade_len_median = spatial_coverage_hull = 0.0
                fixation_grid_entropy = transition_entropy = 0.0
                unique_grid_cells = 0

            target_box = WALDO_BOXES[aoi_name]
            target_cx = (target_box[0] + target_box[2]) / 2.0
            target_cy = (target_box[1] + target_box[3]) / 2.0
            if total_fix > 0:
                target_distances = np.sqrt((fix_x.to_numpy(dtype=float) - target_cx) ** 2 +
                                           (fix_y.to_numpy(dtype=float) - target_cy) ** 2)
                mean_distance_to_target = float(target_distances.mean())
                min_distance_to_target = float(target_distances.min())
            else:
                mean_distance_to_target = min_distance_to_target = 0.0

            waldo_hits = len(waldo_df)
            if waldo_hits > 0:
                waldo_dur_total = waldo_df["duration_ms"].sum() / 1000.0 if "duration_ms" in waldo_df.columns else 0.0
                waldo_dur_avg = waldo_dur_total / waldo_hits
                waldo_first_fixation = waldo_df["timestamp_s"].min()
                waldo_last_fixation = waldo_df["timestamp_s"].max()
                waldo_revisits = waldo_df[["x", "y"]].round(2).duplicated().sum()
                peripheral_rate = len(waldo_df[waldo_df["type"] == "peripheral"]) / waldo_hits
                direct_rate = len(waldo_df[waldo_df["type"] == "direct"]) / waldo_hits
                first_hit_type = waldo_df.sort_values("timestamp_s").iloc[0]["type"]
                ttff_waldo_s = float(max(0.0, waldo_first_fixation - start_s))
                verification_time_s = float(max(0.0, end_s - waldo_first_fixation))
            else:
                waldo_dur_total = waldo_dur_avg = waldo_first_fixation = waldo_last_fixation = waldo_revisits = peripheral_rate = 0.0
                direct_rate = 0.0
                first_hit_type = "none"
                ttff_waldo_s = np.nan
                verification_time_s = np.nan

            gaze_entropy = gaze_entropy(
                gaze_df["gaze position on surface x [normalized]"].to_numpy(dtype=float),
                gaze_df["gaze position on surface y [normalized]"].to_numpy(dtype=float),
            ) if not gaze_df.empty else 0.0

            rec_data = {
                "recording_id": rec_id,
                "level_name": aoi_name,
                "level_duration_s": level_duration_s,
                "total_fixations": total_fix,
                "avg_fixation_duration_s": avg_fix_dur,
                "total_fixation_duration_s": total_duration,
                "saccade_length_avg": saccade_len,
                "saccade_length_median": saccade_len_median,
                "scanpath_length_total": scanpath_length_total,
                "scanpath_length_per_s": scanpath_length_total / level_duration_s if level_duration_s > 0 else 0.0,
                "fixation_dispersion": fixation_dispersion,
                "fixation_revisits": revisits,
                "spatial_coverage_hull": spatial_coverage_hull,
                "fixation_grid_entropy_4x4": fixation_grid_entropy,
                "transition_entropy_4x4": transition_entropy,
                "unique_grid_cells_4x4": unique_grid_cells,
                "mean_distance_to_target": mean_distance_to_target,
                "min_distance_to_target": min_distance_to_target,
                "waldo_fixations": waldo_hits,
                "waldo_fixation_duration_total_s": waldo_dur_total,
                "waldo_fixation_duration_avg_s": waldo_dur_avg,
                "waldo_fixation_ratio": waldo_hits / total_fix if total_fix > 0 else 0.0,
                "waldo_time_ratio": waldo_dur_total / total_duration if total_duration > 0 else 0.0,
                "waldo_first_fixation_s": waldo_first_fixation,
                "waldo_last_fixation_s": waldo_last_fixation,
                "waldo_revisits": waldo_revisits,
                "ttff_waldo_s": ttff_waldo_s,
                "ttff_waldo_ratio": ttff_waldo_s / level_duration_s if waldo_hits > 0 and level_duration_s > 0 else np.nan,
                "verification_time_after_first_hit_s": verification_time_s,
                "verification_time_ratio": verification_time_s / level_duration_s if waldo_hits > 0 and level_duration_s > 0 else np.nan,
                "direct_waldo_ratio": direct_rate,
                "first_hit_type": first_hit_type,
                "gaze_entropy": gaze_entropy,
                "surface_visibility_ratio_global": global_vis_ratio,
                "surface_gaze_ratio_global": global_gaze_ratio,
                "fixation_density": total_fix / vis_frames if vis_frames > 0 else 0.0,
                "fixation_rate_per_s": total_fix / level_duration_s if level_duration_s > 0 else 0.0,
                "peripheral_gaze_rate": peripheral_rate
            }

            summary.append(rec_data)
            pd.DataFrame([rec_data]).to_csv(os.path.join(rec_out_dir, f"features_{rec_id}_{aoi_name}.csv"), index=False)

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, "features_summary.csv"), index=False)
        print("Feature extraction complete.")
