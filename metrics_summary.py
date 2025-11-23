import os
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
from scipy.stats import pearsonr

recordings_dir = "recordings"
waldo_dir = "waldo_fixations"
saliency_dir = "saliency_metrics"
out_dir = "metrics_outputs"
os.makedirs(out_dir, exist_ok=True)

MAP_W, MAP_H = 224, 224
GAUSS_SIG = 6

def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except:
        return pd.DataFrame()

def make_fix_map(xs, ys, durations, w=MAP_W, h=MAP_H, sigma=GAUSS_SIG):
    im = np.zeros((h, w), dtype=float)
    if len(xs) == 0:
        return im
    px = np.clip((xs*(w-1)).astype(int), 0, w-1)
    py = np.clip((ys*(h-1)).astype(int), 0, h-1)
    for x, y, d in zip(px, py, durations):
        im[y, x] += d
    im = gaussian_filter(im, sigma=sigma)
    if im.max() > 0:
        im /= im.max()
    return im

def load_saliency_map(rec_id):
    # DeepGaze/SAM map
    npy = os.path.join(saliency_dir, f"{rec_id}_saliency_map.npy")
    if os.path.exists(npy):
        s = np.load(npy)
        s = cv2.resize(s, (MAP_W, MAP_H), interpolation=cv2.INTER_LINEAR)
        s = s - s.min()
        if s.max() > 0:
            s = s / s.max()
        return s.astype(np.float32)
    # fallback: SpectralResidual
    png = os.path.join(saliency_dir, f"{rec_id}_saliency_vis.png")
    if os.path.exists(png):
        s = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        s = cv2.resize(s, (MAP_W, MAP_H), interpolation=cv2.INTER_LINEAR)
        s = s - s.min()
        if s.max() > 0:
            s = s / s.max()
        return s.astype(np.float32)
    return np.zeros((MAP_H, MAP_W), dtype=np.float32)

def nss_metric(saliency, fix_points):
    if len(fix_points)==0: return np.nan
    if saliency.std() < 1e-12: return 0.0
    s = (saliency - saliency.mean())/saliency.std()
    vals = [s[int(np.clip(y*(MAP_H-1),0,MAP_H-1)), int(np.clip(x*(MAP_W-1),0,MAP_W-1))] for x,y in fix_points]
    return float(np.mean(vals))

def pearson_map(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12: return np.nan
    return float(pearsonr(a.ravel(), b.ravel())[0])

def kl_divergence(p_map, q_map):
    p = np.clip(p_map.ravel()/ (p_map.sum()+1e-9), 1e-12, 1.0)
    q = np.clip(q_map.ravel()/ (q_map.sum()+1e-9), 1e-12, 1.0)
    return float(np.sum(p*np.log(p/q)))

def auc_judd(saliency, fix_points, num_rand=1000):
    if len(fix_points)==0: return np.nan
    s = (saliency - saliency.min())/(saliency.max()-saliency.min()+1e-9)
    ths = np.linspace(0,1,100)
    fix_mask = np.zeros_like(s, dtype=np.uint8)
    for x, y in fix_points:
        fix_mask[int(np.clip(y*(MAP_H-1),0,MAP_H-1)), int(np.clip(x*(MAP_W-1),0,MAP_W-1))] = 1
    fix_coords = np.column_stack(np.where(fix_mask==1))
    all_coords = np.column_stack(np.where(np.ones_like(s)))
    np.random.seed(0)
    rand_idx = np.random.choice(all_coords.shape[0], size=min(num_rand, all_coords.shape[0]), replace=False)
    rand_coords = all_coords[rand_idx]
    tp, fp = [], []
    for th in ths:
        bin_map = (s>=th).astype(np.uint8)
        tp.append(bin_map[fix_coords[:,0], fix_coords[:,1]].sum()/len(fix_coords))
        fp.append(bin_map[rand_coords[:,0], rand_coords[:,1]].sum()/len(rand_coords))
    return float(auc(fp,tp))


summary = []

for rec_id in sorted(os.listdir(recordings_dir)):
    rec_path = os.path.join(recordings_dir, rec_id)
    if not os.path.isdir(rec_path): continue

    fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
    waldo_file = os.path.join(waldo_dir, f"waldo_fixations_{rec_id}.csv")
    fix_df = safe_read_csv(fix_file)
    waldo_df = safe_read_csv(waldo_file)

    if not fix_df.empty:
        xs, ys = fix_df["fixation x [normalized]"].to_numpy(), fix_df["fixation y [normalized]"].to_numpy()
        ds = fix_df.get("duration [ms]", pd.Series(0.2)).to_numpy()/1000.0
        fix_map = make_fix_map(xs, ys, ds)
        fix_points = list(zip(xs, ys))
    else:
        fix_map = np.zeros((MAP_H,MAP_W),dtype=float)
        fix_points = []

    if not waldo_df.empty:
        xs_w, ys_w = waldo_df["x"].to_numpy(), waldo_df["y"].to_numpy()
        ds_w = waldo_df.get("duration_ms", pd.Series(0.2)).to_numpy()/1000.0
        waldo_map = make_fix_map(xs_w, ys_w, ds_w)
        waldo_points = list(zip(xs_w, ys_w))
    else:
        waldo_map = np.zeros((MAP_H,MAP_W),dtype=float)
        waldo_points = []

    sal_map = load_saliency_map(rec_id)

    metrics = {
        "recording_id": rec_id,
        "num_fixations_total": len(fix_points),
        "num_fixations_waldo": len(waldo_points),
        "NSS_all_vs_saliency": nss_metric(sal_map, fix_points),
        "NSS_waldo_vs_saliency": nss_metric(sal_map, waldo_points),
        "KL_fixmap_saliency": kl_divergence(fix_map+1e-12, sal_map+1e-12),
        "Pearson_fixmap_saliency": pearson_map(fix_map, sal_map),
        "AUCJ_fix_saliency": auc_judd(sal_map, fix_points),
        "AUCJ_waldo_saliency": auc_judd(sal_map, waldo_points)
    }
    summary.append(metrics)

    # for name, m in [("fix", fix_map), ("waldo", waldo_map), ("saliency", sal_map)]:
    #     overlay = ((m - m.min())/(m.max()-m.min()+1e-9)*255).astype(np.uint8)
    #     overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    #     out_png = os.path.join(out_dir, f"{rec_id}_overlay_{name}.png")
    #     cv2.imwrite(out_png, overlay)
    # overlays
    for name, m in [("fix", fix_map), ("waldo", waldo_map), ("saliency", sal_map)]:
        overlay_map = gaussian_filter(m, sigma=3)
        overlay = (overlay_map * 255).astype(np.uint8)
        overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        out_png = os.path.join(out_dir, f"{rec_id}_overlay_{name}.png")
        cv2.imwrite(out_png, overlay)

pd.DataFrame(summary).to_csv(os.path.join(out_dir,"metrics_summary.csv"),index=False)
