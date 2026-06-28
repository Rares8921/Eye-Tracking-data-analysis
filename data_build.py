import os
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter

from utils import (
    ATTENTION_MAPS_DIR,
    DATASET_IMAGES_DIR,
    RECORDINGS_DIR,
    waldo_coords as WALDO_BOXES,
    WALDO_DIR,
    recordings as WALDO_INTERVALS,
    waldo_levels as WALDO_TO_IMAGE,
    add_relative_time_columns
)


def make_fixation_map(xs, ys, durations, w=224, h=224, sigma=6):
    im = np.zeros((h, w), dtype=float)
    if len(xs) == 0:
        return im

    px = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((ys * (h - 1)).astype(int), 0, h - 1)

    for x, y, d in zip(px, py, durations):
        im[y, x] += d

    im = gaussian_filter(im, sigma=sigma)
    if im.max() > 0:
        im /= im.max()
    return im


def make_kde_map(xs, ys, w=224, h=224, bandwidth=0.04):
    if len(xs) == 0:
        return np.zeros((h, w))

    xy = np.vstack([xs, ys]).T
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(xy)

    gx, gy = np.linspace(0, 1, w), np.linspace(0, 1, h)
    X, Y = np.meshgrid(gx, gy)
    sample_grid = np.vstack([X.ravel(), Y.ravel()]).T

    log_d = kde.score_samples(sample_grid)
    dens = np.exp(log_d).reshape((h, w))
    dens = gaussian_filter(dens, sigma=2)

    if dens.max() > 0:
        dens /= dens.max()
    return dens


def draw_scale_bar(orig_h, orig_w, cmap_cv):
    bar_w = int(orig_w * 0.02)
    text_w = int(orig_w * 0.05)
    scale_img = np.full((orig_h, bar_w + text_w, 3), 30, dtype=np.uint8)

    gradient = np.linspace(255, 0, orig_h).astype(np.uint8)
    gradient = np.repeat(gradient[:, np.newaxis], bar_w, axis=1)
    scale_img[:, :bar_w] = cv2.applyColorMap(gradient, cmap_cv)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, orig_h / 1000.0)
    cv2.putText(scale_img, "Max", (bar_w + 5, int(orig_h * 0.05)), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(scale_img, "Min", (bar_w + 5, int(orig_h * 0.98)), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return scale_img


def overlay_map_high_res(map_img, resolved_path, out_path, alpha=0.5, cmap_cv=cv2.COLORMAP_JET):
    if not resolved_path:
        return
    bg_img = cv2.imread(resolved_path)
    if bg_img is None:
        return

    orig_h, orig_w = bg_img.shape[:2]
    map_resized = cv2.resize(map_img, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    map_uint8 = (map_resized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(map_uint8, cmap_cv)
    mask = map_uint8 > 2

    overlay = bg_img.copy()
    blended = cv2.addWeighted(bg_img, 1 - alpha, heatmap, alpha, 0)
    overlay[mask] = blended[mask]

    scale_img = draw_scale_bar(orig_h, orig_w, cmap_cv)
    final_img = np.hstack((overlay, scale_img))

    cv2.imwrite(out_path, final_img)


def process_clusters(features, xs, ys, durations, eps, min_samples):
    if len(features) < min_samples:
        return np.full(len(xs), -1), pd.DataFrame()

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(features).labels_

    clusters_data = []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        mask = labels == lab
        clusters_data.append({
            "cluster": int(lab),
            "centroid_x": float(xs[mask].mean()),
            "centroid_y": float(ys[mask].mean()),
            "n_fixations": int(mask.sum()),
            "sum_duration_s": float(durations[mask].sum())
        })

    return labels, pd.DataFrame(clusters_data)


if __name__ == "__main__":
    os.makedirs(str(ATTENTION_MAPS_DIR), exist_ok=True)

    for rec_id in os.listdir(str(RECORDINGS_DIR)):
        rec_path = os.path.join(str(RECORDINGS_DIR), rec_id)
        if not os.path.isdir(rec_path):
            continue

        fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
        waldo_file = os.path.join(str(WALDO_DIR), f"waldo_fixations_{rec_id}.csv")

        if not os.path.exists(fix_file):
            continue

        full_fix_df = pd.read_csv(fix_file)
        if full_fix_df.empty:
            continue

        full_waldo_df = pd.read_csv(waldo_file) if os.path.exists(waldo_file) else pd.DataFrame()

        add_relative_time_columns(
            full_fix_df,
            start_col="start timestamp [ns]",
            end_col="end timestamp [ns]",
            duration_col="duration [ms]"
        )

        rec_out_dir = os.path.join(str(ATTENTION_MAPS_DIR), rec_id)
        os.makedirs(rec_out_dir, exist_ok=True)

        intervals = WALDO_INTERVALS.get(rec_id, [])
        for aoi_name, start_s, end_s in intervals:
            base_out_path = os.path.join(rec_out_dir, f"{rec_id}_{aoi_name}")
            img_filename = WALDO_TO_IMAGE.get(aoi_name, "")

            resolved_img_path = os.path.join(str(DATASET_IMAGES_DIR), img_filename)
            if not os.path.exists(resolved_img_path):
                resolved_img_path = resolved_img_path.replace('.png', '.jpg')
                if not os.path.exists(resolved_img_path):
                    resolved_img_path = ""

            if resolved_img_path:
                ref_img_hd = cv2.imread(resolved_img_path)
            else:
                ref_img_hd = np.zeros((1080, 1920, 3), dtype=np.uint8)

            cv2.imwrite(f"{base_out_path}_reference.png", ref_img_hd)

            mask_fix = (full_fix_df["start_s"] >= start_s) & (full_fix_df["start_s"] <= end_s)
            fix_df = full_fix_df[mask_fix].copy()

            fix_x = fix_df["fixation x [normalized]"].to_numpy(
                dtype=float) if "fixation x [normalized]" in fix_df.columns else np.array([])
            fix_y = fix_df["fixation y [normalized]"].to_numpy(
                dtype=float) if "fixation y [normalized]" in fix_df.columns else np.array([])
            fix_dur = fix_df["duration [ms]"].to_numpy(
                dtype=float) / 1000.0 if "duration [ms]" in fix_df.columns else np.full(len(fix_x), 0.2)

            waldo_df = pd.DataFrame()
            if not full_waldo_df.empty and "waldo" in full_waldo_df.columns:
                waldo_df = full_waldo_df[full_waldo_df["waldo"] == aoi_name].copy()

            w_x, w_y, w_dur = np.array([]), np.array([]), np.array([])
            if not waldo_df.empty:
                x_col = "x" if "x" in waldo_df.columns else "fixation x [normalized]"
                y_col = "y" if "y" in waldo_df.columns else "fixation y [normalized]"
                dur_col = "duration_ms" if "duration_ms" in waldo_df.columns else "duration [ms]"
                t_col = "timestamp_s" if "timestamp_s" in waldo_df.columns else "start_s"

                w_x = waldo_df[x_col].to_numpy(dtype=float)
                w_y = waldo_df[y_col].to_numpy(dtype=float)
                w_dur = waldo_df[dur_col].to_numpy(dtype=float) / 1000.0 if dur_col in waldo_df.columns else np.full(
                    len(w_x), 0.2)

                t = waldo_df[t_col].to_numpy(dtype=float) if t_col in waldo_df.columns else np.arange(len(w_x),
                                                                                                      dtype=float)
                t_norm = (t - t.min()) / (t.max() - t.min()) if len(t) > 0 and t.max() > t.min() else np.zeros_like(t)

                types_code = np.zeros_like(w_x, dtype=float)
                if "type" in waldo_df.columns:
                    types_code = np.array(
                        [0 if str(s).lower().strip().startswith("direct") else 1 for s in waldo_df["type"]],
                        dtype=float)

                feat = np.vstack([w_x, w_y, t_norm * 0.2, types_code * 0.1]).T
                labels, stats_df = process_clusters(feat, w_x, w_y, w_dur, 0.15, 2)

                waldo_df["cluster"] = labels
                waldo_df.to_csv(f"{base_out_path}_waldo_clusters.csv", index=False)
                stats_df.to_csv(f"{base_out_path}_waldo_cluster_stats.csv", index=False)
            else:
                pd.DataFrame().to_csv(f"{base_out_path}_waldo_clusters.csv", index=False)
                pd.DataFrame().to_csv(f"{base_out_path}_waldo_cluster_stats.csv", index=False)

            if not fix_df.empty and aoi_name in WALDO_BOXES:
                x_min, y_min, x_max, y_max = WALDO_BOXES[aoi_name]
                margin = 0.0125
                x_min_p, x_max_p = max(0.0, x_min - margin), min(1.0, x_max + margin)
                y_min_p, y_max_p = max(0.0, y_min - margin), min(1.0, y_max + margin)

                distractor_mask = ~((fix_x >= x_min_p) & (fix_x <= x_max_p) & (fix_y >= y_min_p) & (fix_y <= y_max_p))
                distractor_df = fix_df.loc[distractor_mask].copy()

                if not distractor_df.empty:
                    d_x = distractor_df["fixation x [normalized]"].to_numpy(dtype=float)
                    d_y = distractor_df["fixation y [normalized]"].to_numpy(dtype=float)
                    d_dur = distractor_df["duration [ms]"].to_numpy(dtype=float) / 1000.0

                    feat = np.vstack([d_x, d_y]).T
                    labels, stats_df = process_clusters(feat, d_x, d_y, d_dur, 0.05, 2)

                    distractor_df["cluster"] = labels
                    distractor_df.to_csv(f"{base_out_path}_distractor_clusters.csv", index=False)
                    stats_df.to_csv(f"{base_out_path}_distractor_cluster_stats.csv", index=False)
                else:
                    pd.DataFrame().to_csv(f"{base_out_path}_distractor_clusters.csv", index=False)
                    pd.DataFrame().to_csv(f"{base_out_path}_distractor_cluster_stats.csv", index=False)
            else:
                pd.DataFrame().to_csv(f"{base_out_path}_distractor_clusters.csv", index=False)
                pd.DataFrame().to_csv(f"{base_out_path}_distractor_cluster_stats.csv", index=False)

            all_map = make_fixation_map(fix_x, fix_y, fix_dur)
            kde_map = make_kde_map(fix_x, fix_y)
            w_map = make_fixation_map(w_x, w_y, w_dur)

            np.save(f"{base_out_path}_map_all.npy", all_map.astype(np.float32))
            np.save(f"{base_out_path}_map_kde.npy", kde_map.astype(np.float32))
            np.save(f"{base_out_path}_map_waldo.npy", w_map.astype(np.float32))

            overlay_map_high_res(kde_map, resolved_img_path, f"{base_out_path}_overlay_kde.png",
                                 cmap_cv=cv2.COLORMAP_MAGMA)
            overlay_map_high_res(all_map, resolved_img_path, f"{base_out_path}_overlay_all.png",
                                 cmap_cv=cv2.COLORMAP_HOT)
            overlay_map_high_res(w_map, resolved_img_path, f"{base_out_path}_overlay_waldo.png",
                                 cmap_cv=cv2.COLORMAP_INFERNO)

    print("Data building complet!")