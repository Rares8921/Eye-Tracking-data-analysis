import os
import pandas as pd
import numpy as np

recordings_dir = "recordings"
waldo_dir = "waldo_fixations"
output_dir = "features"
os.makedirs(output_dir, exist_ok=True)

def duration_s(row):
    return (row["end timestamp [ns]"] - row["start timestamp [ns]"]) / 1e9

summary = []

for rec_id in os.listdir(recordings_dir):
    rec_path = os.path.join(recordings_dir, rec_id)
    if not os.path.isdir(rec_path):
        continue

    fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
    gaze_file = os.path.join(rec_path, "gaze_positions_on_surface_Surface 1.csv")
    surf_vis_file = os.path.join(rec_path, "surface_visibility.csv")
    surf_gaze_dist_file = os.path.join(rec_path, "surface_gaze_distribution.csv")
    waldo_file = os.path.join(waldo_dir, f"waldo_fixations_{rec_id}.csv")

    if not os.path.exists(fix_file) or not os.path.exists(gaze_file) or not os.path.exists(waldo_file):
        continue

    fix_df = pd.read_csv(fix_file)
    gaze_df = pd.read_csv(gaze_file)
    waldo_df = pd.read_csv(waldo_file)

    fix_df["duration_s"] = fix_df.apply(duration_s, axis=1)
    total_fix = len(fix_df)
    total_duration = fix_df["duration_s"].sum()
    avg_fix_dur = fix_df["duration_s"].mean() if total_fix > 0 else 0

    if total_fix > 1:
        centroid_shift = np.sqrt(np.diff(fix_df["fixation x [normalized]"])**2 + np.diff(fix_df["fixation y [normalized]"])**2)
        fix_df = fix_df.iloc[1:].copy()
        fix_df["centroid_shift"] = centroid_shift
        saccade_len = fix_df["centroid_shift"].mean()
    else:
        saccade_len = 0

    fixation_dispersion = fix_df[["fixation x [normalized]", "fixation y [normalized]"]].std().mean() if total_fix > 0 else 0
    revisits = fix_df[["fixation x [normalized]", "fixation y [normalized]"]].round(2).duplicated().sum() if total_fix > 1 else 0

    waldo_hits = len(waldo_df)
    waldo_dur_total = waldo_df["duration_ms"].sum() / 1000 if "duration_ms" in waldo_df else 0
    waldo_dur_avg = waldo_dur_total / waldo_hits if waldo_hits > 0 else 0
    fix_ratio_waldo = waldo_hits / total_fix if total_fix > 0 else 0
    waldo_time_ratio = waldo_dur_total / total_duration if total_duration > 0 else 0
    waldo_first_fixation = waldo_df["timestamp_s"].min() if waldo_hits > 0 else 0
    waldo_last_fixation = waldo_df["timestamp_s"].max() if waldo_hits > 0 else 0
    waldo_revisits = waldo_df[["x","y"]].round(2).duplicated().sum() if waldo_hits > 1 else 0

    gaze_entropy = 0
    if len(gaze_df) > 0:
        hist, _, _ = np.histogram2d(
            gaze_df["gaze position on surface x [normalized]"],
            gaze_df["gaze position on surface y [normalized]"],
            bins=30, range=[[0,1],[0,1]]
        )
        p = hist.flatten() / hist.sum()
        p = p[p > 0]
        gaze_entropy = -np.sum(p * np.log2(p))

    surf_vis_ratio = 0
    visible_frames = 0
    try:
        surf_vis = pd.read_csv(surf_vis_file, skiprows=1)
        total_frames = int(pd.read_csv(surf_vis_file, nrows=1).iloc[0,1])
        visible_frames = int(surf_vis.iloc[0,1])
        surf_vis_ratio = visible_frames / total_frames if total_frames > 0 else 0
    except:
        with open(surf_vis_file, "r") as f:
            lines = f.readlines()
            try:
                total_frames = int(lines[0].strip().split(",")[1])
                visible_frames = int(lines[2].strip().split(",")[1])
                surf_vis_ratio = visible_frames / total_frames
            except:
                surf_vis_ratio = 0
                visible_frames = 0

    surf_gaze_ratio = 0
    try:
        surf_gaze = pd.read_csv(surf_gaze_dist_file, skiprows=1)
        total_gaze = int(pd.read_csv(surf_gaze_dist_file, nrows=1).iloc[0,1])
        surface_gaze = int(surf_gaze.iloc[0,1])
        surf_gaze_ratio = surface_gaze / total_gaze if total_gaze > 0 else 0
    except:
        with open(surf_gaze_dist_file,"r") as f:
            lines = f.readlines()
            try:
                total_gaze = int(lines[0].strip().split(",")[1])
                surface_gaze = int(lines[2].strip().split(",")[1])
                surf_gaze_ratio = surface_gaze / total_gaze if total_gaze > 0 else 0
            except:
                surf_gaze_ratio = 0

    fixation_density = total_fix / visible_frames if visible_frames > 0 else 0
    peripheral_gaze_rate = len(waldo_df[waldo_df["type"]=="peripheral"]) / waldo_hits if waldo_hits > 0 else 0

    rec_data = {
        "recording_id": rec_id,
        "total_fixations": total_fix,
        "avg_fixation_duration_s": avg_fix_dur,
        "total_fixation_duration_s": total_duration,
        "saccade_length_avg": saccade_len,
        "fixation_dispersion": fixation_dispersion,
        "fixation_revisits": revisits,
        "waldo_fixations": waldo_hits,
        "waldo_fixation_duration_total_s": waldo_dur_total,
        "waldo_fixation_duration_avg_s": waldo_dur_avg,
        "waldo_fixation_ratio": fix_ratio_waldo,
        "waldo_time_ratio": waldo_time_ratio,
        "waldo_first_fixation_s": waldo_first_fixation,
        "waldo_last_fixation_s": waldo_last_fixation,
        "waldo_revisits": waldo_revisits,
        "gaze_entropy": gaze_entropy,
        "surface_visibility_ratio": surf_vis_ratio,
        "surface_gaze_ratio": surf_gaze_ratio,
        "fixation_density": fixation_density,
        "peripheral_gaze_rate": peripheral_gaze_rate
    }

    summary.append(rec_data)
    out_path = os.path.join(output_dir, f"features_{rec_id}.csv")
    pd.DataFrame([rec_data]).to_csv(out_path, index=False)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(output_dir, "features_summary.csv"), index=False)
