import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

recordings_dir = "recordings"
waldo_dir = "waldo_fixations"
out_dir = "attention_maps"
os.makedirs(out_dir, exist_ok=True)

MAP_W = 224
MAP_H = 224
GAUSSIAN_SIGMA_PIX = 6
KDE_BANDWIDTH = 0.04
DBSCAN_EPS = 0.06
DBSCAN_MIN_SAMPLES = 2
TEMPORAL_SCALE = 0.5
TYPE_SCALE = 0.3

def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def make_fixation_map(xs, ys, durations, w=MAP_W, h=MAP_H, sigma=GAUSSIAN_SIGMA_PIX):
    im = np.zeros((h, w), dtype=float)
    if len(xs) == 0:
        return im

    px = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((ys * (h - 1)).astype(int), 0, h - 1)

    for x, y, d in zip(px, py, durations):
        im[y, x] += d

    im = gaussian_filter(im, sigma=sigma)
    if im.max() > 0:
        im = im / im.max()

    return im

def make_kde_map(xs, ys, w=MAP_W, h=MAP_H, bandwidth=KDE_BANDWIDTH):
    if len(xs) == 0:
        return np.zeros((h, w))

    xy = np.vstack([xs, ys]).T
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(xy)

    gx = np.linspace(0, 1, w)
    gy = np.linspace(0, 1, h)
    X, Y = np.meshgrid(gx, gy)
    sample_grid = np.vstack([X.ravel(), Y.ravel()]).T

    log_d = kde.score_samples(sample_grid)
    dens = np.exp(log_d).reshape((h, w))
    dens = gaussian_filter(dens, sigma=2)

    if dens.max() > 0:
        dens = dens / dens.max()

    return dens

def overlay_map_on_image(map_img, image_path, out_path, alpha=0.6, cmap="hot"):
    try:
        import imageio
        from PIL import Image

        img = imageio.imread(image_path)
        bg = Image.fromarray(img).convert("RGBA")

        heat = (plt.cm.get_cmap(cmap)(map_img) * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat).convert("RGBA")
        heat_img = heat_img.resize(bg.size, resample=Image.BILINEAR)

        blended = Image.blend(bg, heat_img, alpha=alpha)
        blended.save(out_path)
        return True
    except:
        return False

for rec_id in os.listdir(recordings_dir):
    rec_path = os.path.join(recordings_dir, rec_id)
    if not os.path.isdir(rec_path):
        continue

    fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
    waldo_file = os.path.join(waldo_dir, f"waldo_fixations_{rec_id}.csv")

    fix_df = safe_read_csv(fix_file)
    waldo_df = safe_read_csv(waldo_file)

    if fix_df.empty and waldo_df.empty:
        continue

    if not fix_df.empty:
        if "fixation x [normalized]" in fix_df.columns and "fixation y [normalized]" in fix_df.columns:
            fix_x = fix_df["fixation x [normalized]"].to_numpy(dtype=float)
            fix_y = fix_df["fixation y [normalized]"].to_numpy(dtype=float)
            if "duration [ms]" in fix_df.columns:
                fix_dur = fix_df["duration [ms]"].to_numpy(dtype=float) / 1000.0
            else:
                fix_dur = np.ones_like(fix_x, dtype=float) * 0.2
        else:
            fix_x = np.array([])
            fix_y = np.array([])
            fix_dur = np.array([])
    else:
        fix_x = np.array([])
        fix_y = np.array([])
        fix_dur = np.array([])

    if not waldo_df.empty:
        if "x" in waldo_df.columns and "y" in waldo_df.columns:
            w_x = waldo_df["x"].to_numpy(dtype=float)
            w_y = waldo_df["y"].to_numpy(dtype=float)
        elif "fixation x [normalized]" in waldo_df.columns and "fixation y [normalized]" in waldo_df.columns:
            w_x = waldo_df["fixation x [normalized]"].to_numpy(dtype=float)
            w_y = waldo_df["fixation y [normalized]"].to_numpy(dtype=float)
        else:
            w_x = np.array([])
            w_y = np.array([])

        if "duration_ms" in waldo_df.columns:
            w_dur = waldo_df["duration_ms"].to_numpy(dtype=float) / 1000.0
        elif "duration [ms]" in waldo_df.columns:
            w_dur = waldo_df["duration [ms]"].to_numpy(dtype=float) / 1000.0
        else:
            w_dur = np.ones_like(w_x, dtype=float) * 0.2

        if "timestamp_s" in waldo_df.columns:
            t = waldo_df["timestamp_s"].to_numpy(dtype=float)
        elif "start_s" in waldo_df.columns:
            t = waldo_df["start_s"].to_numpy(dtype=float)
        else:
            t = np.arange(len(w_x), dtype=float)

        if t.max() > 0:
            t_norm = (t - t.min()) / (t.max() - t.min())
        else:
            t_norm = np.zeros_like(t)

        if "type" in waldo_df.columns:
            types = waldo_df["type"].astype(str).to_numpy()
            types_code = np.array([0 if s.lower().strip().startswith("direct") else 1 for s in types], dtype=float)
        else:
            types_code = np.zeros_like(w_x, dtype=float)

    else:
        w_x = np.array([])
        w_y = np.array([])
        w_dur = np.array([])
        t_norm = np.array([])
        types_code = np.array([])

    combined_x = np.concatenate([fix_x, w_x])
    combined_y = np.concatenate([fix_y, w_y])
    combined_dur = np.concatenate([fix_dur, w_dur])

    all_map = make_fixation_map(combined_x, combined_y, combined_dur, w=MAP_W, h=MAP_H, sigma=GAUSSIAN_SIGMA_PIX)
    kde_map = make_kde_map(combined_x, combined_y, w=MAP_W, h=MAP_H, bandwidth=KDE_BANDWIDTH)
    w_map = make_fixation_map(w_x, w_y, w_dur, w=MAP_W, h=MAP_H, sigma=GAUSSIAN_SIGMA_PIX)

    np.save(os.path.join(out_dir, f"{rec_id}_map_all.npy"), all_map.astype(np.float32))
    np.save(os.path.join(out_dir, f"{rec_id}_map_kde.npy"), kde_map.astype(np.float32))
    np.save(os.path.join(out_dir, f"{rec_id}_map_waldo.npy"), w_map.astype(np.float32))

    plt.figure(figsize=(6,6))
    plt.imshow(all_map, origin="lower", cmap="hot", extent=[0,1,0,1])
    plt.title(f"{rec_id} attention map (all fixations)")
    plt.axis("off")
    plt.colorbar(label="normalized intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rec_id}_map_all.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.imshow(kde_map, origin="lower", cmap="magma", extent=[0,1,0,1])
    plt.title(f"{rec_id} KDE map (all fixations)")
    plt.axis("off")
    plt.colorbar(label="density")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rec_id}_map_kde.png"))
    plt.close()

    plt.figure(figsize=(6,6))
    plt.imshow(w_map, origin="lower", cmap="inferno", extent=[0,1,0,1])
    plt.title(f"{rec_id} waldo fixation map")
    plt.axis("off")
    plt.colorbar(label="normalized intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{rec_id}_map_waldo.png"))
    plt.close()

    if len(w_x) > 0:
        feat = np.vstack([w_x, w_y, t_norm * TEMPORAL_SCALE, types_code * TYPE_SCALE]).T

        if len(feat) >= 2:
            clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(feat)
            labels = clustering.labels_
        else:
            labels = np.array([-1] * len(w_x))

        waldo_df["cluster"] = labels

        clusters = []
        for lab in np.unique(labels):
            if lab == -1:
                continue
            mask = labels == lab
            cx = w_x[mask].mean()
            cy = w_y[mask].mean()
            total_d = w_dur[mask].sum()
            nfix = mask.sum()
            clusters.append({
                "cluster": int(lab),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "n_fixations": int(nfix),
                "sum_duration_s": float(total_d)
            })

        clusters_df = pd.DataFrame(clusters)

        waldo_df.to_csv(os.path.join(out_dir, f"{rec_id}_waldo_clusters.csv"), index=False)
        clusters_df.to_csv(os.path.join(out_dir, f"{rec_id}_waldo_cluster_stats.csv"), index=False)

    else:
        pd.DataFrame().to_csv(os.path.join(out_dir, f"{rec_id}_waldo_clusters.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(out_dir, f"{rec_id}_waldo_cluster_stats.csv"), index=False)

    image_candidates = ["image.png", "surface_image.png", "frame.png"]
    image_path = None

    for nm in image_candidates:
        p = os.path.join(rec_path, nm)
        if os.path.exists(p):
            image_path = p
            break

    if image_path:
        overlay_map_on_image(kde_map, image_path, os.path.join(out_dir, f"{rec_id}_overlay_kde.png"), alpha=0.5, cmap="magma")
        overlay_map_on_image(all_map, image_path, os.path.join(out_dir, f"{rec_id}_overlay_all.png"), alpha=0.5, cmap="hot")
        overlay_map_on_image(w_map, image_path, os.path.join(out_dir, f"{rec_id}_overlay_waldo.png"), alpha=0.5, cmap="inferno")
