import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError

ROOT_DIR = Path(__file__).resolve().parent

ATTENTION_MAPS_DIR = ROOT_DIR / "attention_maps"
CLASSICAL_MODELS_OUTPUTS = ROOT_DIR / "classic_outputs"
DATASET_IMAGES_DIR = ROOT_DIR / "dataset" / "test" / "images"
EDA_DIR = ROOT_DIR / "eda"
EXPLAINABILITY_DIR = ROOT_DIR / "explainability"
EXTRA_FIGURES_DIR = ROOT_DIR / "extra_figures"
FEATURES_DIR = ROOT_DIR / "features"
SALIENCY_DIR = ROOT_DIR / "saliency_metrics"
SEQUENCE_MODELS_OUTPUTS = ROOT_DIR / "sequence_outputs"
SENSITIVITY_DIR = ROOT_DIR / "sensitivity"
RECORDINGS_DIR = ROOT_DIR / "recordings"
TRANSITION_MODELS_OUTPUTS = ROOT_DIR / "transition_outputs"
WALDO_DIR = ROOT_DIR / "waldo_fixations"

LEVELS = ["waldo_1", "waldo_2", "waldo_3", "waldo_4"]
GRID_SIZE = 4
SEED = 9340

recordings = {
    "2025-09-26_16-24-13-61db42a9": [
        ("waldo_3", 0.000, 60.785),
        ("waldo_4", 60.892, 96.252)
    ],
    "2025-09-26_16-45-00-95032692": [
        ("waldo_1", 0.000, 98.074),
        ("waldo_2", 101.917, 112.556),
        ("waldo_3", 116.497, 121.522),
        ("waldo_4", 123.295, 137.777)
    ],
    "2025-09-26_19-01-53-2ec86210": [
        ("waldo_1", 0.000, 311.217),
        ("waldo_2", 314.024, 325.354),
        ("waldo_3", 329.393, 419.438),
        ("waldo_4", 423.477, 451.358)
    ],
    "2025-09-26_19-10-19-05411873": [
        ("waldo_1", 6.995, 86.055),
        ("waldo_2", 87.976, 91.868),
        ("waldo_3", 95.611, 99.404),
        ("waldo_4", 102.901, 118.861)
    ],
    "2025-09-26_19-12-44-115b6a40": [
        ("waldo_1", 8.340, 136.545),
        ("waldo_2", 138.663, 140.240),
        ("waldo_3", 144.525, 189.794),
        ("waldo_4", 192.306, 193.735)
    ],
    "2025-09-26_19-17-52-c0419fea": [
        ("waldo_1", 0.000, 17.733),
        ("waldo_2", 18.226, 35.368),
        ("waldo_3", 38.964, 62.460),
        ("waldo_4", 66.499, 69.209)
    ],
    "2025-09-26_19-27-31-2699d4a1": [
        ("waldo_1", 0.000, 60.736),
        ("waldo_2", 62.608, 76.203),
        ("waldo_3", 80.735, 97.040),
        ("waldo_4", 99.503, 107.187)
    ],
    "2025-09-26_19-29-58-b4f7a1df": [
        ("waldo_1", 15.467, 130.388),
        ("waldo_2", 132.752, 136.102),
        ("waldo_3", 140.190, 146.348),
        ("waldo_4", 150.436, 152.801)
    ],
    "2025-09-26_19-37-31-91230d39": [
        ("waldo_1", 31.526, 64.775),
        ("waldo_2", 66.844, 68.962),
        ("waldo_3", 71.031, 77.632),
        ("waldo_4", 81.178, 84.429)
    ],
    "2025-09-26_19-39-21-36589e44": [
        ("waldo_1", 5.369, 148.860),
        ("waldo_2", 155.608, 189.400),
        ("waldo_3", 193.390, 349.638),
        ("waldo_4", 352.791, 443.821)
    ],
    "2025-09-26_20-20-21-410567f0": [
        ("waldo_1", 21.772, 117.679),
        ("waldo_2", 120.339, 124.428),
        ("waldo_3", 126.841, 159.549),
        ("waldo_4", 163.145, 255.012)
    ]
}

waldo_coords = {
    "waldo_1": (931/1979, 375/1079, 974/1979, 433/1079),
    "waldo_2": (1197/1979, 186/1079, 1277/1979, 382/1079),
    "waldo_3": (446/1979, 204/1079, 484/1979, 263/1079),
    "waldo_4": (1081/1979, 306/1079, 1107/1979, 397/1079)
}

waldo_levels = {
    "waldo_1": "level1.png",
    "waldo_2": "level2.png",
    "waldo_3": "level3.png",
    "waldo_4": "level4.png",
}


def add_relative_time_columns(
    df: pd.DataFrame,
    start_col: str,
    end_col: str | None = None,
    duration_col: str | None = None,
    start_out: str = "start_s",
    end_out: str = "end_s",
) -> bool:
    # True if duration was used
    if df.empty:
        df[start_out] = pd.Series(dtype=float)
        df[end_out] = pd.Series(dtype=float)
        return False

    start_ns = pd.to_numeric(df.get(start_col), errors="coerce")
    origin = start_ns.dropna().iloc[0] if start_ns.notna().any() else np.nan

    if np.isfinite(origin) and start_ns.nunique(dropna=True) > 1:
        df[start_out] = (start_ns - origin) / 1e9

        if end_col and end_col in df.columns:
            end_ns = pd.to_numeric(df[end_col], errors="coerce")
            if end_ns.nunique(dropna=True) > 1:
                df[end_out] = (end_ns - origin) / 1e9
            elif duration_col and duration_col in df.columns:
                df[end_out] = df[start_out] + pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0) / 1000.0

        elif duration_col and duration_col in df.columns:
            df[end_out] = df[start_out] + pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0) / 1000.0

    elif duration_col and duration_col in df.columns:
        duration_s = pd.to_numeric(df[duration_col], errors="coerce").fillna(0.0) / 1000.0
        df[start_out] = duration_s.shift(fill_value=0.0).cumsum()
        df[end_out] = df[start_out] + duration_s
        return True

    else:
        df[start_out] = 0.0
        df[end_out] = 0.0

    return False


def parse_neon_summary_csv(
        path: str | Path,
        surface_name: str = "Surface 1"
) -> tuple[int, int]:
    # (gaze on surface counts, total gaze counts)
    path = Path(path)

    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return 0, 0

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return 0, 0

    total_count = 0
    header_row = [col.strip() for col in lines[0].split(",")]
    if len(header_row) >= 2:
        total_count = int(float(header_row[1]))

    visible_count = 0
    for line in lines[2:]:
        row = [col.strip() for col in line.split(",")]

        if len(row) >= 2 and row[0] == surface_name:
            visible_count = int(float(row[1]))
            break

    return visible_count, total_count


def fixation_timestamp_is_truncated(filepath: str | Path) -> bool:
    path = Path(filepath)
    if not path.exists():
        return False

    try:
        with path.open("r", encoding="utf-8") as handle:
            next(handle, None)
            for _ in range(5):
                line = next(handle, "")
                parts = line.strip().split(",")
                if len(parts) >= 3 and "E+" in parts[1]:
                    return True
    except OSError:
        return False

    return False


def participant_id_from_recording(recording_id: str) -> str:
    return recording_id.split("-")[-1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(torch_module=None) -> None:
    np.random.seed(SEED)
    if torch_module is not None:
        torch_module.manual_seed(SEED)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(SEED)


def convex_hull_area(xs, ys) -> float:
    if len(xs) < 3:
        return 0.0
    try:
        points = np.column_stack([xs, ys])
        hull = ConvexHull(points)
        return float(hull.volume)
    except QhullError:
        return 0.0


def grid_cell_ids(xs, ys, grid_size: int = GRID_SIZE) -> np.ndarray:
    if len(xs) == 0:
        return np.array([], dtype=int)

    xs = np.asarray(xs)
    ys = np.asarray(ys)

    x_bins = (xs * grid_size).astype(int)
    y_bins = (ys * grid_size).astype(int)

    x_bins = np.clip(x_bins, 0, grid_size - 1)
    y_bins = np.clip(y_bins, 0, grid_size - 1)

    # row-major flatten
    cell_ids = y_bins * grid_size + x_bins
    return cell_ids


def cell_entropy(cell_ids, grid_size: int = GRID_SIZE) -> float:
    if len(cell_ids) == 0:
        return 0.0

    total_cells = grid_size * grid_size
    counts = np.bincount(cell_ids, minlength=total_cells).astype(float)

    visited = counts[counts > 0]
    probs = visited / counts.sum()

    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def transition_entropy(cell_ids, grid_size: int = GRID_SIZE) -> float:
    # 2 fixations = 1 transition
    if len(cell_ids) < 2:
        return 0.0

    state_count = grid_size * grid_size
    transition_counts = np.zeros((state_count, state_count), dtype=float)

    # cnt from cell src to cell dst
    for src, dst in zip(cell_ids[:-1], cell_ids[1:]):
        transition_counts[src, dst] += 1.0

    row_totals = transition_counts.sum(axis=1)
    total_transitions = row_totals.sum()

    if total_transitions <= 0:
        return 0.0

    entropy = 0.0
    for src in range(state_count):
        if row_totals[src] <= 0:
            continue
        # Weight row by how often we were in this state
        p_state = row_totals[src] / total_transitions
        row_probs = transition_counts[src, transition_counts[src] > 0] / row_totals[src]
        entropy -= p_state * np.sum(row_probs * np.log2(row_probs))

    return float(entropy)


def gaze_entropy(xs, ys, bins: int = 30) -> float:
    if len(xs) == 0:
        return 0.0

    hist, _, _ = np.histogram2d(xs, ys, bins=bins, range=[[0, 1], [0, 1]])

    counts = hist.flatten()
    total = counts.sum()

    if total <= 0:
        return 0.0

    probs = counts[counts > 0] / total
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)



def binary_f1(
    y_true: list[int],
    y_pred: list[int]
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) > 0:
        return float(2 * precision * recall / (precision + recall))
    return 0.0


def macro_f1(
    y_true: list[int],
    y_pred: list[int],
    n_classes: int
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    scores = []
    for cls in range(n_classes):
        tp = float(((y_true == cls) & (y_pred == cls)).sum())
        fp = float(((y_true != cls) & (y_pred == cls)).sum())
        fn = float(((y_true == cls) & (y_pred != cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores.append(f1)

    return float(np.mean(scores))


def balanced_accuracy(
    y_true: list[int],
    y_pred: list[int],
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    recalls = []
    for cls in [0, 1]:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recall = float((y_pred[mask] == cls).mean())
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def balanced_accuracy_multiclass(
    y_true: list[int],
    y_pred: list[int],
    n_classes: int
) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    recalls = []
    for cls in range(n_classes):
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        recall = float((y_pred[mask] == cls).mean())
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0

def saliency_values(
    xs,
    ys,
    saliency_map: np.ndarray
) -> np.ndarray:
    if saliency_map is None or len(xs) == 0:
        return np.zeros(len(xs), dtype=np.float32)

    h, w = saliency_map.shape
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    # normalized coords -> pixel indices
    px = np.clip((xs * (w - 1)).astype(int), 0, w - 1)
    py = np.clip((ys * (h - 1)).astype(int), 0, h - 1)

    return saliency_map[py, px].astype(np.float32)


def saliency_maps() -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for level in LEVELS:
        path = SALIENCY_DIR / f"global_{level}_saliency_map.npy"
        if path.exists():
            maps[level] = np.load(path)

    return maps


def long_search_labels(
    rows_df: pd.DataFrame
) -> pd.DataFrame:
    df = rows_df.copy()

    # median search time
    medians = df.groupby("level_name")["total_search_time_s"].median()
    medians = medians.rename("level_median_search_s")
    df = df.merge(medians, on="level_name", how="left")

    df["label_long_search"] = (df["total_search_time_s"] > df["level_median_search_s"]).astype(int)
    return df
