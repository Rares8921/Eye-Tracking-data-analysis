import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import (
    EXPLAINABILITY_DIR as OUTPUT_DIR,
    CLASSICAL_MODELS_OUTPUTS,
    RECORDINGS_DIR,
    recordings as WALDO_INTERVALS,
    add_relative_time_columns,
    balanced_accuracy,
    saliency_maps as SALIENCY_MAPS,
    saliency_values,
    set_seed
)

WINDOW_S = 5.0
MAX_SEQ_LEN = 60

gaze_only_cols = [
    "early_fix_count", "early_fix_rate", "early_avg_fix_duration_s", "early_total_fix_duration_s",
    "early_saccade_length_avg", "early_saccade_length_median", "early_scanpath_length_total",
    "early_scanpath_length_per_s",
    "early_spatial_coverage_hull", "early_fixation_entropy_4x4", "early_transition_entropy_4x4",
    "early_unique_grid_cells_4x4",
    "early_mean_distance_to_target", "early_min_distance_to_target", "early_gaze_entropy", "early_gaze_count",
    "early_mean_saliency", "early_max_saliency"
]

feature_sets = {
    "gaze_only": gaze_only_cols,
    "gaze_plus_target": gaze_only_cols + ["early_waldo_hit_count", "early_direct_hit_count",
                                          "early_peripheral_hit_count", "early_hit_any"]
}

TASK_CONFIGS = [
    ("label_found_waldo", "gaze_plus_target"),
    ("label_long_search", "gaze_only"),
]

FEATURE_GROUPS = {
    "gaze_dynamics": ["early_fix_count", "early_fix_rate", "early_avg_fix_duration_s", "early_total_fix_duration_s",
                      "early_saccade_length_avg", "early_saccade_length_median", "early_scanpath_length_total",
                      "early_scanpath_length_per_s"],
    "spatial_search": ["early_spatial_coverage_hull", "early_unique_grid_cells_4x4", "early_mean_distance_to_target",
                       "early_min_distance_to_target"],
    "entropy_structure": ["early_fixation_entropy_4x4", "early_transition_entropy_4x4", "early_gaze_entropy",
                          "early_gaze_count"],
    "saliency_alignment": ["early_mean_saliency", "early_max_saliency"],
    "target_contact": ["early_waldo_hit_count", "early_direct_hit_count", "early_peripheral_hit_count",
                       "early_hit_any"],
}


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, lengths, return_attention=False):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True)
        mask = torch.arange(out.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

        attn_scores = self.attention(out).squeeze(-1).masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * out, dim=1)
        logits = self.head(context).squeeze(-1)

        if return_attention:
            return logits, attn_weights, mask
        return logits


def train_linear_model(x_train, y_train, epochs=180, lr=1e-2):
    set_seed(torch)
    model = nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pos_weight = torch.tensor([float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)],
                              dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t)
        loss.backward()
        optimizer.step()

    return model


def evaluate_logreg_grouped(dataset_df, feature_columns, label_column):
    feature_df = dataset_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = pd.to_numeric(dataset_df[label_column], errors="coerce")
    valid_mask = labels.notna()

    x = feature_df.loc[valid_mask].to_numpy(dtype=float)
    y = labels.loc[valid_mask].astype(int).to_numpy()
    groups = dataset_df.loc[valid_mask, "participant_id"].astype(str).to_numpy()

    unique_groups = list(dict.fromkeys(groups.tolist()))
    all_true, all_pred = [], []

    for held_out_group in unique_groups:
        test_idx = np.where(groups == held_out_group)[0]
        train_idx = np.where(groups != held_out_group)[0]
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2: continue

        x_train, x_test = x[train_idx], x[test_idx]
        mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True) + 1e-8

        model = train_linear_model((x_train - mean) / std, y_train, epochs=180)

        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(torch.tensor((x_test - mean) / std, dtype=torch.float32)).squeeze(
                1)) >= 0.5).cpu().numpy().astype(int)

        all_true.extend(y[test_idx].tolist())
        all_pred.extend(preds.tolist())

    return balanced_accuracy(all_true, all_pred) if all_true else np.nan


def build_weight_table(dataset_df, feature_columns, label_column):
    feature_df = dataset_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = pd.to_numeric(dataset_df[label_column], errors="coerce")
    valid_mask = labels.notna()
    x = feature_df.loc[valid_mask].to_numpy(dtype=float)
    y = labels.loc[valid_mask].astype(int).to_numpy()

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8

    model = train_linear_model((x - mean) / std, y, epochs=260)
    weights = model.weight.detach().cpu().numpy().reshape(-1)

    return pd.DataFrame({
        "feature": feature_columns,
        "weight": weights,
        "abs_weight": np.abs(weights),
        "direction": np.where(weights >= 0, "positive", "negative"),
    }).sort_values("abs_weight", ascending=False)


def run_group_ablation(dataset_df):
    rows = []
    for task_name, feature_set_name in TASK_CONFIGS:
        feature_columns = list(feature_sets[feature_set_name])
        baseline_score = evaluate_logreg_grouped(dataset_df, feature_columns, task_name)
        rows.append({"task": task_name, "feature_set": feature_set_name, "ablation": "none",
                     "balanced_accuracy": baseline_score, "score_drop": 0.0})

        for group_name, group_features in FEATURE_GROUPS.items():
            kept_features = [f for f in feature_columns if f not in group_features]
            if not kept_features: continue
            score = evaluate_logreg_grouped(dataset_df, kept_features, task_name)
            rows.append({
                "task": task_name, "feature_set": feature_set_name, "ablation": group_name,
                "balanced_accuracy": score,
                "score_drop": baseline_score - score if np.isfinite(score) and np.isfinite(baseline_score) else np.nan,
            })
    return pd.DataFrame(rows)


def train_sequence_attention_summary(dataset_df):
    s_maps = SALIENCY_MAPS()
    padded_sequences, lengths, labels = [], [], []

    for row in dataset_df.itertuples():
        rec_id = getattr(row, "recording_id", None)
        level_name = getattr(row, "level_name", None)
        label = getattr(row, "label_long_search", np.nan)

        intervals = WALDO_INTERVALS.get(rec_id, [])
        start_s = next((s for lvl, s, e in intervals if lvl == level_name), None)
        if start_s is None or pd.isna(label): continue

        fix_path = RECORDINGS_DIR / rec_id / "fixations_on_surface_Surface 1.csv"
        if not fix_path.exists(): continue

        fix_df = pd.read_csv(fix_path)
        add_relative_time_columns(fix_df, start_col="start timestamp [ns]", end_col="end timestamp [ns]",
                                  duration_col="duration [ms]")

        l_fix = fix_df[(fix_df["start_s"] >= start_s) & (fix_df["start_s"] <= start_s + WINDOW_S)].copy()
        if len(l_fix) < 3: continue

        xs, ys = l_fix["fixation x [normalized]"].to_numpy(dtype=float), l_fix["fixation y [normalized]"].to_numpy(
            dtype=float)
        durs = pd.to_numeric(l_fix["duration [ms]"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0

        dx, dy = np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0])
        saccade_len = np.sqrt(dx ** 2 + dy ** 2)
        velocity = saccade_len / (durs + 1e-5)
        saccade_angle = np.arctan2(dy, dx) / np.pi

        seq = np.stack([
            xs, ys, durs, saccade_len, saccade_angle, velocity,
            np.diff(velocity, prepend=velocity[0]),
            np.diff(saccade_angle, prepend=saccade_angle[0]),
            saliency_values(xs, ys, s_maps.get(level_name))
        ], axis=1).astype(np.float32)

        seq_len = len(seq)
        seq = seq[:MAX_SEQ_LEN] if seq_len >= MAX_SEQ_LEN else np.vstack(
            [seq, np.zeros((MAX_SEQ_LEN - seq_len, 9), dtype=np.float32)])

        padded_sequences.append(seq)
        lengths.append(min(seq_len, MAX_SEQ_LEN))
        labels.append(label)

    if not padded_sequences:
        return pd.DataFrame()

    x = np.array(padded_sequences)
    seq_lengths = np.array(lengths)
    y = np.array(labels).astype(int)

    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True) + 1e-8
    x = (x - mean) / std

    set_seed(torch)
    model = SequenceClassifier(input_dim=x.shape[2], hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    pos_weight = torch.tensor([float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    len_tensor = torch.tensor(seq_lengths, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model.train()
    for _ in range(70):
        optimizer.zero_grad()
        loss = criterion(model(x_tensor, len_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, attn, mask = model(x_tensor, len_tensor, return_attention=True)
        attn, mask = attn.cpu().numpy(), mask.cpu().numpy()

    rows = []
    for pos_idx in range(attn.shape[1]):
        valid = mask[:, pos_idx]
        if valid.sum() == 0: continue
        values = attn[valid, pos_idx]
        rows.append({
            "position_index": int(pos_idx), "mean_attention": float(values.mean()),
            "median_attention": float(np.median(values)), "n_sequences": int(valid.sum()),
        })
    return pd.DataFrame(rows)


def write_summary(ablation_df, weight_tables, attention_df):
    lines = ["# Explainability Summary\n"]
    if not ablation_df.empty:
        lines.extend(["## Group Ablation\n", ablation_df.to_markdown(index=False), "\n"])

    for (task_name, feature_set_name), weight_df in weight_tables.items():
        lines.extend(
            [f"## Linear Weights: {task_name} / {feature_set_name}\n", weight_df.head(12).to_markdown(index=False),
             "\n"])

    if not attention_df.empty:
        lines.extend(["## Sequence Attention Summary\n", attention_df.to_markdown(index=False), "\n"])

    (OUTPUT_DIR / "explainability_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset_path = CLASSICAL_MODELS_OUTPUTS / f"early_window_dataset_{int(WINDOW_S)}s.csv"
    if not os.path.exists(dataset_path):
        print(f"Eroare: Nu exista fisierul {dataset_path}.")
        exit()

    dataset_df = pd.read_csv(dataset_path)

    ablation_df = run_group_ablation(dataset_df)
    ablation_df.to_csv(OUTPUT_DIR / "feature_group_ablation.csv", index=False)

    weight_tables = {}
    for task_name, feature_set_name in TASK_CONFIGS:
        weight_df = build_weight_table(dataset_df, list(feature_sets[feature_set_name]), task_name)
        weight_tables[(task_name, feature_set_name)] = weight_df
        weight_df.to_csv(OUTPUT_DIR / f"weights_{task_name}_{feature_set_name}.csv", index=False)

    attention_df = train_sequence_attention_summary(dataset_df)
    attention_df.to_csv(OUTPUT_DIR / "sequence_attention_summary.csv", index=False)

    write_summary(ablation_df, weight_tables, attention_df)