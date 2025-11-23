import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.metrics import auc
from scipy.stats import pearsonr
import cv2

recordings_dir = "recordings"
attention_dir = "attention_maps"
saliency_dir = "saliency" #saliency_metrics
out_dir = "saliency_metrics"
os.makedirs(out_dir, exist_ok=True)

MAP_W = 224
MAP_H = 224
GAUSS_SIG = 6

def safe_read_csv(p):
    try:
        return pd.read_csv(p)
    except:
        return None

def make_fix_map_from_fixations(fix_file, w=MAP_W, h=MAP_H):
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
    im = np.zeros((h, w), dtype=float)
    px = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((ys * (h - 1)).astype(int), 0, h - 1)
    for x, y, d in zip(px, py, ds):
        im[y, x] += d
    im = gaussian_filter(im, sigma=GAUSS_SIG)
    if im.max() > 0:
        im = im / im.max()
    fix_points = list(zip(xs.tolist(), ys.tolist()))
    return im.astype(np.float32), fix_points

def load_attention_map(rec_id):
    p = os.path.join(attention_dir, f"{rec_id}_map_all.npy")
    if os.path.exists(p):
        m = np.load(p)
        if m.shape != (MAP_H, MAP_W):
            m = cv2.resize(m, (MAP_W, MAP_H))
        if m.max() > 0:
            m = m.astype(np.float32) / m.max()
        return m
    return None

def compute_saliency_from_image(img, w=MAP_W, h=MAP_H):
    if img is None:
        return np.zeros((h, w), dtype=np.float32)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        _, salmap = sal.computeSaliency(img)
        salmap = salmap.astype(np.float32)
        salmap = cv2.resize(salmap, (w, h))
    except:
        grayf = cv2.resize(gray.astype(np.float32), (w, h))
        dft = cv2.dft(grayf, flags=cv2.DFT_COMPLEX_OUTPUT)
        mag, ang = cv2.cartToPolar(dft[:,:,0], dft[:,:,1])
        logmag = np.log(mag + 1e-9)
        avg = cv2.GaussianBlur(logmag, (3,3), 0)
        spectral = logmag - avg
        exp_spec = np.exp(spectral)
        salmap = (exp_spec - exp_spec.min()) / (exp_spec.max() - exp_spec.min() + 1e-9)
    salmap = gaussian_filter(salmap, sigma=2)
    if salmap.max() > 0:
        salmap = salmap / salmap.max()
    return salmap.astype(np.float32)

def load_reference_image(rec_path):
    candidates = ["surface_image.png", "image.png", "frame.png", "world.png", "world.jpg", "world_frame.png"]
    for nm in candidates:
        p = os.path.join(rec_path, nm)
        if os.path.exists(p):
            im = cv2.imread(p)
            if im is not None:
                im = cv2.resize(im, (MAP_W, MAP_H))
                return im
    mp4 = os.path.join(rec_path, "world.mp4")
    if os.path.exists(mp4):
        cap = cv2.VideoCapture(mp4)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        cap.release()
        if ok:
            frame = cv2.resize(frame, (MAP_W, MAP_H))
            return frame
    return None

def nss_metric(saliency, fix_points):
    if len(fix_points) == 0:
        return np.nan
    s = saliency.copy().astype(np.float32)
    s = (s - s.mean()) / (s.std() + 1e-9)
    vals = []
    for x, y in fix_points:
        xi = int(np.clip(x * (s.shape[1]-1), 0, s.shape[1]-1))
        yi = int(np.clip(y * (s.shape[0]-1), 0, s.shape[0]-1))
        vals.append(s[yi, xi])
    return float(np.mean(vals))

def kl_divergence(p_map, q_map):
    p = p_map.astype(np.float64).ravel()
    q = q_map.astype(np.float64).ravel()
    p = p / (p.sum() + 1e-9)
    q = q / (q.sum() + 1e-9)
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))

def pearson_map(a, b):
    if a.ravel().std() < 1e-12 or b.ravel().std() < 1e-12:
        return np.nan
    return float(pearsonr(a.ravel(), b.ravel())[0])

def auc_judd(saliency, fix_points, num_rand=1000):
    if len(fix_points) == 0:
        return np.nan
    s = saliency.copy().astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    ths = np.linspace(0, 1, 100)
    fix_mask = np.zeros_like(s, dtype=np.uint8)
    for x, y in fix_points:
        xi = int(np.clip(x * (s.shape[1]-1), 0, s.shape[1]-1))
        yi = int(np.clip(y * (s.shape[0]-1), 0, s.shape[0]-1))
        fix_mask[yi, xi] = 1
    fix_coords = np.column_stack(np.where(fix_mask == 1))
    num_fix = fix_coords.shape[0]
    all_coords = np.column_stack(np.where(np.ones_like(s)))
    np.random.seed(0)
    rand_idx = np.random.choice(all_coords.shape[0], size=min(num_rand, all_coords.shape[0]), replace=False)
    rand_coords = all_coords[rand_idx]
    tp = []
    fp = []
    for th in ths:
        bin_map = (s >= th).astype(np.uint8)
        tpr = bin_map[fix_coords[:,0], fix_coords[:,1]].sum() / float(num_fix)
        fpr = bin_map[rand_coords[:,0], rand_coords[:,1]].sum() / float(len(rand_coords))
        tp.append(tpr)
        fp.append(fpr)
    try:
        return float(auc(fp, tp))
    except:
        return np.nan

summary = []

for rec_id in sorted(os.listdir(recordings_dir)):
    rec_path = os.path.join(recordings_dir, rec_id)
    if not os.path.isdir(rec_path):
        continue
    att_map = load_attention_map(rec_id)
    if att_map is None:
        fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
        att_map, _ = make_fix_map_from_fixations(fix_file)
    waldo_fix_file = os.path.join("waldo_fixations", f"waldo_fixations_{rec_id}.csv")
    waldo_map, waldo_points = make_fix_map_from_fixations(waldo_fix_file)
    fix_file = os.path.join(rec_path, "fixations_on_surface_Surface 1.csv")
    fix_map, fix_points = make_fix_map_from_fixations(fix_file)
    gaze_map_path = os.path.join(attention_dir, f"{rec_id}_map_kde.npy")
    if os.path.exists(gaze_map_path):
        gaze_map = np.load(gaze_map_path)
        if gaze_map.max() > 0:
            gaze_map = gaze_map.astype(np.float32) / gaze_map.max()
    else:
        gaze_map = att_map.copy()
    sal_map = None
    sal_npy = os.path.join(saliency_dir, f"{rec_id}_saliency.npy")
    sal_png = os.path.join(saliency_dir, f"{rec_id}_saliency.png")
    if os.path.exists(sal_npy):
        s = np.load(sal_npy)
        s = cv2.resize(s, (MAP_W, MAP_H))
        sal_map = s.astype(np.float32)
        if sal_map.max() > 0:
            sal_map = sal_map / sal_map.max()
    elif os.path.exists(sal_png):
        s = cv2.imread(sal_png, cv2.IMREAD_GRAYSCALE)
        if s is not None:
            s = cv2.resize(s, (MAP_W, MAP_H))
            s = s.astype(np.float32)
            if s.max() > 0:
                s = s / s.max()
            sal_map = s
    else:
        ref = load_reference_image(rec_path)
        sal_map = compute_saliency_from_image(ref)
    if sal_map is None:
        sal_map = np.zeros((MAP_H, MAP_W), dtype=np.float32)
    metrics = {}
    metrics["recording_id"] = rec_id
    metrics["num_fixations_total"] = len(fix_points)
    metrics["num_fixations_waldo"] = len(waldo_points)
    metrics["NSS_all_vs_saliency"] = nss_metric(sal_map, fix_points)
    metrics["NSS_waldo_vs_saliency"] = nss_metric(sal_map, waldo_points)
    try:
        metrics["KL_fixmap_saliency"] = kl_divergence(fix_map+1e-12, sal_map+1e-12)
    except:
        metrics["KL_fixmap_saliency"] = np.nan
    try:
        metrics["Pearson_fixmap_saliency"] = pearson_map(fix_map, sal_map)
    except:
        metrics["Pearson_fixmap_saliency"] = np.nan
    metrics["AUCJ_fix_saliency"] = auc_judd(sal_map, fix_points)
    metrics["AUCJ_waldo_saliency"] = auc_judd(sal_map, waldo_points)
    out_csv = os.path.join(out_dir, f"saliency_metrics_{rec_id}.csv")
    pd.DataFrame([metrics]).to_csv(out_csv, index=False)
    summary.append(metrics)
    np.save(os.path.join(out_dir, f"{rec_id}_saliency_map.npy"), sal_map.astype(np.float32))
    overlay = (sal_map * 255).astype(np.uint8)
    overlay = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(out_dir, f"{rec_id}_saliency_vis.png"), overlay)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(out_dir, "saliency_metrics_summary.csv"), index=False)
