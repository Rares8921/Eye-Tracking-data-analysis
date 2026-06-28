import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import pearsonr
from sklearn.metrics import auc

from utils import (
    DATASET_IMAGES_DIR,
    RECORDINGS_DIR,
    SALIENCY_DIR as OUT_DIR,
    WALDO_DIR as WALDO_FIX_DIR,
    recordings as WALDO_INTERVALS,
    waldo_levels as WALDO_TO_IMAGE,
    add_relative_time_columns
)

MAP_W = 224
MAP_H = 224
CS_SCALES = [(1, 3), (1, 5), (2, 5), (2, 7), (3, 7), (3, 9)]

WEIGHTS = {
    "intensity": 1.0,
    "color": 1.0,
    "orientation": 1.0,
    "edge_density": 0.5,
    "center_bias": 0.1,
    "color_popout": 0.4,
}


def _normalize_01(m):
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def itti_normalize(m):
    m = _normalize_01(m)
    if m.max() < 1e-12:
        return m

    radius = max(3, min(m.shape) // 20)
    local_max = maximum_filter(m, size=radius * 2 + 1)
    is_peak = (m == local_max) & (m > 0.05)

    if not is_peak.any():
        return m

    weight = (m.max() - m[is_peak].mean()) ** 2
    return np.clip(m * weight, 0, 1).astype(np.float32)


def compute_center_surround(channel, scales=CS_SCALES):
    result = np.zeros_like(channel, dtype=np.float64)
    for sigma_c, sigma_s in scales:
        center = cv2.GaussianBlur(channel, (0, 0), sigma_c)
        surround = cv2.GaussianBlur(channel, (0, 0), sigma_s)
        cs = np.abs(center.astype(np.float64) - surround.astype(np.float64))
        result += _normalize_01(cs)
    return _normalize_01(result).astype(np.float32)


def compute_color_saliency(img_float):
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    R = np.maximum(0, r - (g + b) / 2.0)
    G = np.maximum(0, g - (r + b) / 2.0)
    B = np.maximum(0, b - (r + g) / 2.0)
    Y = np.maximum(0, (r + g) / 2.0 - np.abs(r - g) / 2.0 - b)

    rg_map, by_map = np.zeros_like(R, dtype=np.float64), np.zeros_like(B, dtype=np.float64)

    for sigma_c, sigma_s in CS_SCALES:
        rg_map += np.abs(cv2.GaussianBlur(R, (0, 0), sigma_c) - cv2.GaussianBlur(G, (0, 0), sigma_s)) + \
                  np.abs(cv2.GaussianBlur(G, (0, 0), sigma_c) - cv2.GaussianBlur(R, (0, 0), sigma_s))
        by_map += np.abs(cv2.GaussianBlur(B, (0, 0), sigma_c) - cv2.GaussianBlur(Y, (0, 0), sigma_s)) + \
                  np.abs(cv2.GaussianBlur(Y, (0, 0), sigma_c) - cv2.GaussianBlur(B, (0, 0), sigma_s))

    return itti_normalize(itti_normalize(_normalize_01(rg_map)) + itti_normalize(_normalize_01(by_map)))


def compute_orientation_saliency(intensity):
    all_orientations = np.zeros_like(intensity, dtype=np.float64)

    for theta_deg in [0, 45, 90, 135]:
        theta_rad = np.radians(theta_deg)
        orientation_map = np.zeros_like(intensity, dtype=np.float64)

        for sigma, wavelength in  [(2, 4), (3, 8), (5, 12)]:
            ksize = int(sigma * 6) | 1
            kern_re = cv2.getGaborKernel((ksize, ksize), sigma, theta_rad, wavelength, 0.5, 0)
            kern_im = cv2.getGaborKernel((ksize, ksize), sigma, theta_rad, wavelength, 0.5, np.pi / 2)

            energy = np.sqrt(cv2.filter2D(intensity, cv2.CV_32F, kern_re) ** 2 +
                             cv2.filter2D(intensity, cv2.CV_32F, kern_im) ** 2)

            for sigma_c, sigma_s in [(1, 3), (2, 5), (3, 7)]:
                orientation_map += np.abs(cv2.GaussianBlur(energy, (0, 0), sigma_c) -
                                          cv2.GaussianBlur(energy, (0, 0), sigma_s))

        all_orientations += itti_normalize(_normalize_01(orientation_map))

    return itti_normalize(all_orientations)


def compute_edge_density(img, sigma_blur=8):
    if img.ndim == 3:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.dtype != np.uint8 else img
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    edges = sum(
        cv2.Canny(gray, low, high).astype(np.float32) / 255.0 for low, high in [(30, 100), (50, 150), (80, 200)]) / 3.0
    return _normalize_01(cv2.GaussianBlur(edges, (0, 0), sigma_blur))


def compute_center_bias(h=MAP_H, w=MAP_W, sigma_frac=0.25):
    y, x = np.ogrid[:h, :w]
    return np.exp(
        -((x - w / 2.0) ** 2 / (2 * (w * sigma_frac) ** 2) + (y - h / 2.0) ** 2 / (2 * (h * sigma_frac) ** 2))).astype(
        np.float32)


def compute_color_popout(img_float):
    hsv = cv2.cvtColor((np.clip(img_float, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    red_mask = (((h_chan < 10) | (h_chan > 170)) & (s_chan > 60) & (v_chan > 50)).astype(np.float32) + \
               (((h_chan < 15) | (h_chan > 165)) & (s_chan > 40) & (v_chan > 80)).astype(np.float32)

    red_map = cv2.GaussianBlur(red_mask / 2.0, (0, 0), 6)
    sat_contrast = compute_center_surround(s_chan.astype(np.float32) / 255.0, scales=[(2, 5), (3, 7)])

    return _normalize_01(_normalize_01(red_map) * 0.6 + _normalize_01(sat_contrast) * 0.4)


def compute_itti_koch_saliency(img):
    img_f = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
    if img_f.ndim == 2:
        img_f = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    h, w = img_f.shape[:2]
    intensity = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    channels = {
        "intensity": itti_normalize(compute_center_surround(intensity)),
        "color": compute_color_saliency(img_f),
        "orientation": compute_orientation_saliency(intensity),
        "edge_density": compute_edge_density(img_f),
        "center_bias": compute_center_bias(h, w),
        "color_popout": compute_color_popout(img_f),
    }

    saliency = sum(WEIGHTS.get(name, 1.0) * ch_map.astype(np.float64) for name, ch_map in channels.items())
    return _normalize_01(gaussian_filter(saliency / sum(WEIGHTS.values()), sigma=3))


if __name__ == "__main__":
    os.makedirs(str(OUT_DIR), exist_ok=True)
    summary = []
    precomputed_saliency = {}

    print("Computing saliency maps")
    for aoi_name, img_filename in WALDO_TO_IMAGE.items():
        img_path = os.path.join(str(DATASET_IMAGES_DIR), img_filename)
        if not os.path.exists(img_path):
            img_path = img_path.replace('.png', '.jpg')

        if os.path.exists(img_path):
            ref_img_hd = cv2.imread(img_path)
            ref_img = cv2.resize(ref_img_hd, (MAP_W, MAP_H))
        else:
            ref_img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

        saliency = compute_itti_koch_saliency(ref_img)
        precomputed_saliency[aoi_name] = saliency

        np.save(os.path.join(str(OUT_DIR), f"global_{aoi_name}_saliency_map.npy"), saliency.astype(np.float32))

        try:
            if os.path.exists(img_path):
                bg_img = cv2.imread(img_path)
                if bg_img is not None:
                    orig_h, orig_w = bg_img.shape[:2]
                    map_resized = cv2.resize(saliency, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                    map_uint8 = (map_resized * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(map_uint8, cv2.COLORMAP_JET)
                    mask = map_uint8 > 2
                    overlay = bg_img.copy()
                    blended = cv2.addWeighted(bg_img, 1 - 0.5, heatmap, 0.5, 0)
                    overlay[mask] = blended[mask]

                    bar_w = int(orig_w * 0.02)
                    text_w = int(orig_w * 0.05)
                    scale_img = np.full((orig_h, bar_w + text_w, 3), 30, dtype=np.uint8)
                    gradient = np.linspace(255, 0, orig_h).astype(np.uint8)
                    gradient = np.repeat(gradient[:, np.newaxis], bar_w, axis=1)
                    scale_img[:, :bar_w] = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.5, orig_h / 1000.0)
                    cv2.putText(scale_img, "Max", (bar_w + 5, int(orig_h * 0.05)), font, font_scale, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    cv2.putText(scale_img, "Min", (bar_w + 5, int(orig_h * 0.98)), font, font_scale, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    final_img = np.hstack((overlay, scale_img))
                    cv2.imwrite(os.path.join(str(OUT_DIR), f"global_{aoi_name}_saliency_overlay.png"), final_img)
        except Exception:
            pass

    if not os.path.exists(str(RECORDINGS_DIR)):
        print(f"Recordings directory '{RECORDINGS_DIR}' not found.")
    else:
        recordings = sorted(
            [d for d in os.listdir(str(RECORDINGS_DIR)) if os.path.isdir(os.path.join(str(RECORDINGS_DIR), d))])
        print(f"Processing {len(recordings)} recordings")

        for rec_id in recordings:
            rec_path = os.path.join(str(RECORDINGS_DIR), rec_id)
            rec_out_dir = os.path.join(str(OUT_DIR), rec_id)
            dir_created = False

            fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
            waldo_file = os.path.join(str(WALDO_FIX_DIR), f"waldo_fixations_{rec_id}.csv")

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
                duration_col="duration [ms]",
            )

            for aoi_name, start_s, end_s in WALDO_INTERVALS.get(rec_id, []):
                fix_df = full_fix_df[(full_fix_df["start_s"] >= start_s) & (full_fix_df["start_s"] <= end_s)].copy()
                waldo_df = full_waldo_df[full_waldo_df[
                                             "waldo"] == aoi_name].copy() if not full_waldo_df.empty and "waldo" in full_waldo_df.columns else pd.DataFrame()

                fix_map, fix_points = np.zeros((MAP_H, MAP_W), dtype=np.float32), []
                if not fix_df.empty:
                    xs = fix_df.get("fixation x [normalized]", fix_df.get("x")).to_numpy(dtype=float)
                    ys = fix_df.get("fixation y [normalized]", fix_df.get("y")).to_numpy(dtype=float)
                    ds = fix_df.get("duration [ms]",
                                    fix_df.get("duration_ms", pd.Series(np.ones_like(xs) * 200))).to_numpy(
                        dtype=float) / 1000.0

                    px, py = np.clip((xs * (MAP_W - 1)).astype(int), 0, MAP_W - 1), np.clip(
                        (ys * (MAP_H - 1)).astype(int), 0, MAP_H - 1)
                    im = np.zeros((MAP_H, MAP_W), dtype=np.float64)
                    for x, y, d in zip(px, py, ds):
                        im[y, x] += d

                    im = gaussian_filter(im, sigma=6)
                    fix_map = (im / im.max()).astype(np.float32) if im.max() > 0 else im.astype(np.float32)
                    fix_points = list(zip(xs, ys))

                waldo_points = []
                if not waldo_df.empty:
                    w_xs = waldo_df.get("fixation x [normalized]", waldo_df.get("x")).to_numpy(dtype=float)
                    w_ys = waldo_df.get("fixation y [normalized]", waldo_df.get("y")).to_numpy(dtype=float)
                    waldo_points = list(zip(w_xs, w_ys))

                if len(fix_points) == 0:
                    continue

                if not dir_created:
                    os.makedirs(rec_out_dir, exist_ok=True)
                    dir_created = True

                saliency = precomputed_saliency[aoi_name]
                center_bias = compute_center_bias(MAP_H, MAP_W)


                def _calc_local_metrics(s, f_map, points, base):
                    if not points: return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    s_f64 = s.astype(np.float64)
                    std = s_f64.std()
                    nss = float(np.mean([(s_f64 - s_f64.mean()) / std[
                        int(np.clip(y * (MAP_H - 1), 0, MAP_H - 1)), int(np.clip(x * (MAP_W - 1), 0, MAP_W - 1))] for
                                         x, y in points])) if std >= 1e-12 else 0.0

                    p_norm = np.clip(f_map.ravel().astype(np.float64), 1e-12, None) / np.clip(
                        f_map.ravel().astype(np.float64), 1e-12, None).sum()
                    q_norm = np.clip(s_f64.ravel(), 1e-12, None) / np.clip(s_f64.ravel(), 1e-12, None).sum()

                    kl = float(np.sum(p_norm * np.log(p_norm / q_norm)))
                    sim = float(np.minimum(p_norm, q_norm).sum())
                    pearson = float(
                        pearsonr(f_map.ravel(), s_f64.ravel())[0]) if std >= 1e-12 and f_map.std() >= 1e-12 else np.nan

                    f_c = np.array(
                        [[int(np.clip(y * (MAP_H - 1), 0, MAP_H - 1)), int(np.clip(x * (MAP_W - 1), 0, MAP_W - 1))] for
                         x, y in points])
                    r_c = np.column_stack(np.where(np.ones_like(s)))
                    r_c = r_c[np.random.RandomState(1235).choice(len(r_c), min(2000, len(r_c)), replace=False)]

                    try:
                        auc_v = float(
                            auc([(s_f64 >= th).astype(np.uint8)[r_c[:, 0], r_c[:, 1]].sum() / len(r_c) for th in
                                 np.linspace(0, 1, 200)],
                                [(s_f64 >= th).astype(np.uint8)[f_c[:, 0], f_c[:, 1]].sum() / len(f_c) for th in
                                 np.linspace(0, 1, 200)]))
                    except:
                        auc_v = np.nan

                    base_p = np.clip(base.ravel().astype(np.float64), 1e-12, None) / np.clip(
                        base.ravel().astype(np.float64), 1e-12, None).sum()
                    f_idx = f_c[:, 0] * MAP_W + f_c[:, 1]
                    ig = float(np.mean(np.log2(q_norm[f_idx]) - np.log2(base_p[f_idx])))
                    return nss, kl, pearson, auc_v, sim, ig


                nss_all, kl_all, pearson_all, auc_all, sim_all, ig_all = _calc_local_metrics(saliency, fix_map,
                                                                                             fix_points, center_bias)
                nss_waldo, _, _, auc_waldo, _, ig_waldo = _calc_local_metrics(saliency, fix_map, waldo_points,
                                                                              center_bias)

                metrics = {
                    "recording_id": rec_id,
                    "level_name": aoi_name,
                    "num_fixations_total": len(fix_points),
                    "num_fixations_waldo": len(waldo_points),
                    "saliency_std": float(saliency.std()),
                    "saliency_range": float(saliency.max() - saliency.min()),
                    "NSS_all_vs_saliency": nss_all,
                    "NSS_waldo_vs_saliency": nss_waldo,
                    "KL_fixmap_saliency": kl_all,
                    "Pearson_fixmap_saliency": pearson_all,
                    "AUCJ_fix_saliency": auc_all,
                    "AUCJ_waldo_saliency": auc_waldo,
                    "SIM_fixmap_saliency": sim_all,
                    "IG_fix_vs_centerbias": ig_all,
                    "IG_waldo_vs_centerbias": ig_waldo,
                }

                summary.append(metrics)
                pd.DataFrame([metrics]).to_csv(os.path.join(rec_out_dir, f"saliency_metrics_{rec_id}_{aoi_name}.csv"),
                                               index=False)

        if summary:
            pd.DataFrame(summary).to_csv(os.path.join(str(OUT_DIR), "saliency_metrics_summary.csv"), index=False)
            print("Saliency complet")