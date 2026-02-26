"""
Compute saliency maps and evaluate how well they predict human fixations.

Uses a multi-channel Itti-Koch model combining intensity, color, orientation,
edge density, center bias, and color pop-out features. Compares predicted
saliency against actual gaze data using NSS, KL-div, Pearson, and AUC-Judd.
"""

import os
import re
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import pearsonr
from sklearn.metrics import auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Config

recordings_dir = "recordings"
attention_dir = "attention_maps"
waldo_fix_dir = "waldo_fixations"
out_dir = "saliency_metrics"
os.makedirs(out_dir, exist_ok=True)

MAP_W = 224
MAP_H = 224

# Model parameters
CS_SCALES = [(1, 3), (1, 5), (2, 5), (2, 7), (3, 7), (3, 9)]
GABOR_ORIENTATIONS = [0, 45, 90, 135]
GABOR_SCALES = [(2, 4), (3, 8), (5, 12)]

WEIGHTS = {
    "intensity": 1.0,
    "color": 1.0,
    "orientation": 1.0,
    "edge_density": 0.5,
    "center_bias": 0.3,
    "color_popout": 0.4,
}


# Reference image extraction

def parse_homography_string(h_str):
    """Parse homography matrix string to 3x3 array."""
    cleaned = re.sub(r'[\[\]\n]', ' ', str(h_str))
    vals = [float(x) for x in cleaned.split()]
    if len(vals) != 9:
        return None
    return np.array(vals).reshape(3, 3)


def extract_surface_from_frame(frame, homography_3x3, out_size=(MAP_W, MAP_H)):
    """Warp video frame to surface coordinates using homography."""
    out_w, out_h = out_size
    S = np.diag([1.0 / out_w, 1.0 / out_h, 1.0])
    M = homography_3x3 @ S
    surface = cv2.warpPerspective(frame, M, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
    return surface


def extract_best_surface_image(rec_path, out_size=(MAP_W, MAP_H)):
    """Extract sharpest well-tracked surface image from video."""
    surf_file = os.path.join(rec_path, "surf_positions_Surface 1.csv")
    video_file = os.path.join(rec_path, "world.mp4")

    if not os.path.exists(surf_file) or not os.path.exists(video_file):
        return _fallback_reference(video_file, out_size)

    try:
        surf_df = pd.read_csv(surf_file)
    except Exception:
        return _fallback_reference(video_file, out_size)

    if surf_df.empty:
        return _fallback_reference(video_file, out_size)

    max_markers = surf_df["num_detected_markers"].max()
    good = surf_df[surf_df["num_detected_markers"] >= max(max_markers - 1, 2)]

    if good.empty:
        return _fallback_reference(video_file, out_size)

    step = max(1, len(good) // 10)
    candidates = good.iloc[::step].head(10)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None

    best_surface = None
    best_sharpness = -1

    for _, row in candidates.iterrows():
        H = parse_homography_string(row["surf_to_img_trans"])
        if H is None:
            continue

        frame_idx = int(row["world_index"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        surface = extract_surface_from_frame(frame, H, out_size)

        gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_surface = surface

    cap.release()

    if best_surface is not None:
        return best_surface

    return _fallback_reference(video_file, out_size)


def _fallback_reference(video_path, out_size):
    """Extract middle frame if homography unavailable."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.resize(frame, out_size)
    return None


# Saliency computation

def _normalize_01(m):
    """Normalize to [0, 1]."""
    mn, mx = m.min(), m.max()
    if mx - mn < 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)


def itti_normalize(m):
    """Normalize and weight by peak strength to emphasize pop-out signals."""
    m = _normalize_01(m)
    if m.max() < 1e-12:
        return m

    radius = max(3, min(m.shape) // 20)
    local_max = maximum_filter(m, size=radius * 2 + 1)
    is_peak = (m == local_max) & (m > 0.05)

    if not is_peak.any():
        return m

    M = m.max()
    m_bar = m[is_peak].mean()

    weight = (M - m_bar) ** 2
    return np.clip(m * weight, 0, 1).astype(np.float32)


def compute_center_surround(channel, scales=None):
    """Multi-scale center-surround using difference of Gaussians."""
    if scales is None:
        scales = CS_SCALES

    result = np.zeros_like(channel, dtype=np.float64)
    for sigma_c, sigma_s in scales:
        center = cv2.GaussianBlur(channel, (0, 0), sigma_c)
        surround = cv2.GaussianBlur(channel, (0, 0), sigma_s)
        cs = np.abs(center.astype(np.float64) - surround.astype(np.float64))
        result += _normalize_01(cs)

    return _normalize_01(result).astype(np.float32)


def compute_intensity_saliency(intensity):
    """Intensity channel: luminance center-surround at multiple scales."""
    return itti_normalize(compute_center_surround(intensity))


def compute_color_saliency(img_float):
    """RG and BY opponent color channels with center-surround."""
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    R = np.maximum(0, r - (g + b) / 2.0)
    G = np.maximum(0, g - (r + b) / 2.0)
    B = np.maximum(0, b - (r + g) / 2.0)
    Y = np.maximum(0, (r + g) / 2.0 - np.abs(r - g) / 2.0 - b)

    rg_map = np.zeros_like(R, dtype=np.float64)
    by_map = np.zeros_like(B, dtype=np.float64)

    for sigma_c, sigma_s in CS_SCALES:
        R_c = cv2.GaussianBlur(R, (0, 0), sigma_c)
        G_c = cv2.GaussianBlur(G, (0, 0), sigma_c)
        R_s = cv2.GaussianBlur(R, (0, 0), sigma_s)
        G_s = cv2.GaussianBlur(G, (0, 0), sigma_s)
        rg_map += np.abs(R_c - G_s) + np.abs(G_c - R_s)

        B_c = cv2.GaussianBlur(B, (0, 0), sigma_c)
        Y_c = cv2.GaussianBlur(Y, (0, 0), sigma_c)
        B_s = cv2.GaussianBlur(B, (0, 0), sigma_s)
        Y_s = cv2.GaussianBlur(Y, (0, 0), sigma_s)
        by_map += np.abs(B_c - Y_s) + np.abs(Y_c - B_s)

    color_map = itti_normalize(_normalize_01(rg_map)) + itti_normalize(_normalize_01(by_map))
    return itti_normalize(color_map)


def compute_orientation_saliency(intensity):
    """Gabor responses at 4 orientations and 3 scales."""
    all_orientations = np.zeros_like(intensity, dtype=np.float64)

    for theta_deg in GABOR_ORIENTATIONS:
        theta_rad = np.radians(theta_deg)
        orientation_map = np.zeros_like(intensity, dtype=np.float64)

        for sigma, wavelength in GABOR_SCALES:
            ksize = int(sigma * 6) | 1
            kern_re = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta_rad, wavelength, 0.5, 0
            )
            kern_im = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta_rad, wavelength, 0.5, np.pi / 2
            )
            resp_re = cv2.filter2D(intensity, cv2.CV_32F, kern_re)
            resp_im = cv2.filter2D(intensity, cv2.CV_32F, kern_im)
            energy = np.sqrt(resp_re ** 2 + resp_im ** 2)

            for sigma_c, sigma_s in [(1, 3), (2, 5), (3, 7)]:
                c = cv2.GaussianBlur(energy, (0, 0), sigma_c)
                s = cv2.GaussianBlur(energy, (0, 0), sigma_s)
                orientation_map += np.abs(c - s)

        all_orientations += itti_normalize(_normalize_01(orientation_map))

    return itti_normalize(all_orientations)


def compute_edge_density(img, sigma_blur=8):
    """Edge density map from multi-threshold Canny."""
    if img.ndim == 3:
        if img.dtype != np.uint8:
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img_u8 = img
        gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    edges1 = cv2.Canny(gray, 30, 100).astype(np.float32) / 255.0
    edges2 = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    edges3 = cv2.Canny(gray, 80, 200).astype(np.float32) / 255.0
    edges = (edges1 + edges2 + edges3) / 3.0

    density = cv2.GaussianBlur(edges, (0, 0), sigma_blur)
    return _normalize_01(density)


def compute_center_bias(h=MAP_H, w=MAP_W, sigma_frac=0.25):
    """Gaussian center bias (humans tend to look near center)."""
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    sigma_x = w * sigma_frac
    sigma_y = h * sigma_frac
    bias = np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) +
                     (y - cy) ** 2 / (2 * sigma_y ** 2)))
    return bias.astype(np.float32)


def compute_color_popout(img_float):
    """Detect red/warm color regions (Waldo's shirt)."""
    img_uint8 = (np.clip(img_float, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2HSV)

    h_chan, s_chan, v_chan = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    red_mask1 = ((h_chan < 10) | (h_chan > 170)) & (s_chan > 60) & (v_chan > 50)
    red_mask2 = ((h_chan < 15) | (h_chan > 165)) & (s_chan > 40) & (v_chan > 80)
    red_density = (red_mask1.astype(np.float32) + red_mask2.astype(np.float32)) / 2.0

    red_map = cv2.GaussianBlur(red_density, (0, 0), 6)

    sat_norm = s_chan.astype(np.float32) / 255.0
    sat_contrast = compute_center_surround(sat_norm, scales=[(2, 5), (3, 7)])

    combined = _normalize_01(red_map) * 0.6 + _normalize_01(sat_contrast) * 0.4
    return _normalize_01(combined)


def compute_itti_koch_saliency(img, weights=None):
    """Compute multi-channel saliency and return combined map + individual channels."""
    if weights is None:
        weights = WEIGHTS

    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img.astype(np.float32)

    if img_f.ndim == 2:
        img_f = cv2.cvtColor(
            (img_f * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        ).astype(np.float32) / 255.0

    h, w = img_f.shape[:2]
    intensity = cv2.cvtColor(
        (img_f * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0

    ch_intensity = compute_intensity_saliency(intensity)
    ch_color = compute_color_saliency(img_f)
    ch_orientation = compute_orientation_saliency(intensity)
    ch_edges = compute_edge_density(img_f)
    ch_center_bias = compute_center_bias(h, w)
    ch_color_popout = compute_color_popout(img_f)

    channels = {
        "intensity": ch_intensity,
        "color": ch_color,
        "orientation": ch_orientation,
        "edge_density": ch_edges,
        "center_bias": ch_center_bias,
        "color_popout": ch_color_popout,
    }

    saliency = np.zeros((h, w), dtype=np.float64)
    total_weight = 0
    for name, ch_map in channels.items():
        wt = weights.get(name, 1.0)
        saliency += wt * ch_map.astype(np.float64)
        total_weight += wt

    saliency /= total_weight

    saliency = gaussian_filter(saliency, sigma=3)
    saliency = _normalize_01(saliency)

    return saliency, channels


# Fixation maps

def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def make_fix_map_from_fixations(fix_file, w=MAP_W, h=MAP_H, sigma=6):
    """Build fixation density map from CSV file."""
    df = safe_read_csv(fix_file)
    if df is None or df.empty:
        return np.zeros((h, w), dtype=np.float32), []

    if "fixation x [normalized]" in df.columns and "fixation y [normalized]" in df.columns:
        xs = df["fixation x [normalized]"].to_numpy(dtype=float)
        ys = df["fixation y [normalized]"].to_numpy(dtype=float)
    elif "x" in df.columns and "y" in df.columns:
        xs = df["x"].to_numpy(dtype=float)
        ys = df["y"].to_numpy(dtype=float)
    else:
        return np.zeros((h, w), dtype=np.float32), []

    if "duration [ms]" in df.columns:
        ds = df["duration [ms]"].to_numpy(dtype=float) / 1000.0
    elif "duration_ms" in df.columns:
        ds = df["duration_ms"].to_numpy(dtype=float) / 1000.0
    else:
        ds = np.ones_like(xs) * 0.2

    im = np.zeros((h, w), dtype=np.float64)
    px = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((ys * (h - 1)).astype(int), 0, h - 1)
    for x, y, d in zip(px, py, ds):
        im[y, x] += d

    im = gaussian_filter(im, sigma=sigma)
    if im.max() > 0:
        im = im / im.max()

    fix_points = list(zip(xs.tolist(), ys.tolist()))
    return im.astype(np.float32), fix_points


# Evaluation metrics

def nss_metric(saliency, fix_points):
    """Normalized Scanpath Saliency - higher means fixations land on salient regions."""
    if len(fix_points) == 0:
        return np.nan
    s = saliency.astype(np.float64)
    std = s.std()
    if std < 1e-12:
        return 0.0
    s_z = (s - s.mean()) / std
    h, w = s.shape
    vals = []
    for x, y in fix_points:
        xi = int(np.clip(x * (w - 1), 0, w - 1))
        yi = int(np.clip(y * (h - 1), 0, h - 1))
        vals.append(s_z[yi, xi])
    return float(np.mean(vals))


def kl_divergence(p_map, q_map):
    """KL divergence between two distributions."""
    p = np.clip(p_map.ravel().astype(np.float64), 1e-12, None)
    q = np.clip(q_map.ravel().astype(np.float64), 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def pearson_cc(a, b):
    """Pearson correlation."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(pearsonr(a.ravel(), b.ravel())[0])


def auc_judd(saliency, fix_points, num_rand=2000):
    """AUC-Judd: ROC curve for fixations vs random locations (0.5=chance, 1.0=perfect)."""
    if len(fix_points) == 0:
        return np.nan

    s = _normalize_01(saliency)
    h, w = s.shape
    thresholds = np.linspace(0, 1, 200)

    fix_mask = np.zeros_like(s, dtype=np.uint8)
    for x, y in fix_points:
        xi = int(np.clip(x * (w - 1), 0, w - 1))
        yi = int(np.clip(y * (h - 1), 0, h - 1))
        fix_mask[yi, xi] = 1

    fix_coords = np.column_stack(np.where(fix_mask == 1))
    if len(fix_coords) == 0:
        return np.nan

    all_coords = np.column_stack(np.where(np.ones_like(s)))
    rng = np.random.RandomState(42)
    rand_idx = rng.choice(len(all_coords), size=min(num_rand, len(all_coords)),
                          replace=False)
    rand_coords = all_coords[rand_idx]

    tpr_list, fpr_list = [], []
    for th in thresholds:
        above = (s >= th).astype(np.uint8)
        tpr = above[fix_coords[:, 0], fix_coords[:, 1]].sum() / len(fix_coords)
        fpr = above[rand_coords[:, 0], rand_coords[:, 1]].sum() / len(rand_coords)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    try:
        return float(auc(fpr_list, tpr_list))
    except Exception:
        return np.nan


# Visualization

def make_saliency_dashboard(rec_id, saliency, channels, fix_map, waldo_map,
                            fix_points, waldo_points, metrics, out_path):
    """Generate interactive HTML dashboard with all maps and metrics."""
    channel_names = list(channels.keys())
    n_channels = len(channel_names)
    n_cols = 3
    n_rows = 1 + (n_channels + n_cols - 1) // n_cols

    titles = ["Combined Saliency + Fixations", "All Fixations Density",
              "Waldo Fixations"]
    titles += [f"Channel: {name}" for name in channel_names]
    while len(titles) < n_rows * n_cols:
        titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles[:n_rows * n_cols],
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    def add_heatmap(data, row, col, colorscale="Inferno"):
        fig.add_trace(
            go.Heatmap(z=data, colorscale=colorscale, showscale=False,
                       hoverinfo="z"),
            row=row, col=col
        )

    # Row 1: main maps
    add_heatmap(saliency, 1, 1, "Inferno")
    add_heatmap(fix_map, 1, 2, "Hot")
    add_heatmap(waldo_map, 1, 3, "Magma")

    if fix_points:
        fx = [p[0] * (MAP_W - 1) for p in fix_points]
        fy = [p[1] * (MAP_H - 1) for p in fix_points]
        fig.add_trace(
            go.Scatter(x=fx, y=fy, mode="markers",
                       marker=dict(size=3, color="cyan", opacity=0.35),
                       name="Fixations", showlegend=False),
            row=1, col=1
        )
    if waldo_points:
        wx = [p[0] * (MAP_W - 1) for p in waldo_points]
        wy = [p[1] * (MAP_H - 1) for p in waldo_points]
        fig.add_trace(
            go.Scatter(x=wx, y=wy, mode="markers",
                       marker=dict(size=5, color="lime", symbol="x"),
                       name="Waldo hits", showlegend=False),
            row=1, col=1
        )

    channel_cmap = {
        "intensity": "Greys",
        "color": "RdYlBu_r",
        "orientation": "Viridis",
        "edge_density": "YlOrRd",
        "center_bias": "Blues",
        "color_popout": "Reds",
    }
    for i, name in enumerate(channel_names):
        r = 2 + i // n_cols
        c = 1 + i % n_cols
        cmap = channel_cmap.get(name, "Viridis")
        add_heatmap(channels[name], r, c, cmap)

    metrics_lines = []
    for k, v in metrics.items():
        if k == "recording_id":
            continue
        if isinstance(v, float):
            metrics_lines.append(f"<b>{k}</b>: {v:.4f}")
        else:
            metrics_lines.append(f"<b>{k}</b>: {v}")
    metrics_text = "<br>".join(metrics_lines)

    fig.update_layout(
        title=dict(text=f"Saliency Analysis: {rec_id}", font=dict(size=16)),
        height=320 * n_rows,
        width=320 * n_cols,
        showlegend=False,
        annotations=list(fig.layout.annotations) + [
            dict(
                text=metrics_text,
                xref="paper", yref="paper",
                x=1.02, y=0.5,
                showarrow=False,
                font=dict(size=10),
                align="left",
                bordercolor="gray",
                borderwidth=1,
                borderpad=6,
                bgcolor="rgba(255,255,255,0.9)",
            )
        ]
    )

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(visible=False, row=r, col=c)
            fig.update_yaxes(visible=False, autorange="reversed", row=r, col=c)

    fig.write_html(out_path)


# Main pipeline

if __name__ == "__main__":
    summary = []

    recordings = sorted([
        d for d in os.listdir(recordings_dir)
        if os.path.isdir(os.path.join(recordings_dir, d))
    ])

    print(f"Processing {len(recordings)} recordings...")
    print(f"Output directory: {out_dir}")
    print()

    for rec_id in recordings:
        rec_path = os.path.join(recordings_dir, rec_id)
        print(f"Recording: {rec_id}")

        print("  Extracting surface image from video...")
        ref_img = extract_best_surface_image(rec_path, out_size=(MAP_W, MAP_H))
        if ref_img is None:
            print("  WARNING: Could not extract reference image, using blank")
            ref_img = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)

        print("  Computing saliency model...")
        saliency, channels = compute_itti_koch_saliency(ref_img)

        sal_std = saliency.std()
        sal_range = saliency.max() - saliency.min()
        print(f"  Saliency stats: range={sal_range:.3f}, std={sal_std:.4f}, "
              f"min={saliency.min():.3f}, max={saliency.max():.3f}")

        fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
        fix_map, fix_points = make_fix_map_from_fixations(fix_file)

        waldo_file = os.path.join(waldo_fix_dir, f"waldo_fixations_{rec_id}.csv")
        waldo_map, waldo_points = make_fix_map_from_fixations(waldo_file)

        print(f"  Fixations: {len(fix_points)} total, {len(waldo_points)} on Waldo")

        metrics = {
            "recording_id": rec_id,
            "num_fixations_total": len(fix_points),
            "num_fixations_waldo": len(waldo_points),
            "saliency_std": float(sal_std),
            "saliency_range": float(sal_range),
            "NSS_all_vs_saliency": nss_metric(saliency, fix_points),
            "NSS_waldo_vs_saliency": nss_metric(saliency, waldo_points),
            "KL_fixmap_saliency": kl_divergence(fix_map + 1e-12, saliency + 1e-12),
            "Pearson_fixmap_saliency": pearson_cc(fix_map, saliency),
            "AUCJ_fix_saliency": auc_judd(saliency, fix_points),
            "AUCJ_waldo_saliency": auc_judd(saliency, waldo_points),
        }

        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        summary.append(metrics)

        np.save(os.path.join(out_dir, f"{rec_id}_saliency_map.npy"),
                saliency.astype(np.float32))

        pd.DataFrame([metrics]).to_csv(
            os.path.join(out_dir, f"saliency_metrics_{rec_id}.csv"), index=False)

        cv2.imwrite(os.path.join(out_dir, f"{rec_id}_reference.png"), ref_img)

        print("  Generating dashboard...")
        make_saliency_dashboard(
            rec_id, saliency, channels, fix_map, waldo_map,
            fix_points, waldo_points, metrics,
            os.path.join(out_dir, f"{rec_id}_saliency_vis.html")
        )

        print()

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, "saliency_metrics_summary.csv"),
                      index=False)
    print(f"Done! Summary saved to {out_dir}/saliency_metrics_summary.csv")
    print()
    print(summary_df.to_string(index=False))
