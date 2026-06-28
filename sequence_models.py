import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from utils import (
    SEED,
    RECORDINGS_DIR,
    SEQUENCE_MODELS_OUTPUTS as OUTPUT_DIR,
    WALDO_DIR,
    recordings as WALDO_INTERVALS,
    add_relative_time_columns,
    long_search_labels,
    balanced_accuracy,
    binary_f1,
    saliency_maps,
    participant_id_from_recording,
    saliency_values,
    set_seed,
)

MAX_SEQ_LEN = 60

def save_figure(path, **kwargs):
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(path, **kwargs)
    plt.savefig(path.with_suffix(".pdf"), **{k: v for k, v in kwargs.items() if k != "dpi"})


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True)
        mask = torch.arange(out.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_scores = self.attention(out).squeeze(-1).masked_fill(~mask, -1e9)
        context = torch.sum(torch.softmax(attn_scores, dim=1).unsqueeze(-1) * out, dim=1)
        return self.head(context).squeeze(-1)


class MeanPoolSequenceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.15),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, x, lengths):
        mask = (torch.arange(x.size(1), device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(-1)
        return self.net((torch.sum(x * mask, dim=1) / lengths.unsqueeze(1).clamp(min=1).float())).squeeze(-1)


def train_and_predict(x_train, len_train, y_train, x_test, len_test, hidden_dim, epochs, model_type):
    set_seed(torch)
    model = SequenceClassifier(x_train.shape[2],
                               hidden_dim) if model_type == "bilstm_attention" else MeanPoolSequenceClassifier(
        x_train.shape[2], 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    pos_weight = torch.tensor([float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)],
                              dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(len_train, dtype=torch.long),
                            torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(dataset, batch_size=min(16, len(x_train)), shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_len, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x, batch_len), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32),
                                    torch.tensor(len_test, dtype=torch.long))) >= 0.5).cpu().numpy().astype(int)


def evaluate_sequence_task(meta_df, sequences, lengths, label_column):
    valid_mask = pd.to_numeric(meta_df[label_column], errors="coerce").notna().to_numpy()
    groups = meta_df.loc[valid_mask, "participant_id"].astype(str).to_numpy()
    y = meta_df.loc[valid_mask, label_column].astype(int).to_numpy()
    x = sequences[valid_mask]
    seq_lengths = lengths[valid_mask]

    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return pd.DataFrame(), pd.DataFrame()

    results, predictions = [], []
    unique_groups = list(dict.fromkeys(groups.tolist()))

    for model_type in ["mean_pool_mlp", "bilstm_attention"]:
        fold_y_true, fold_y_pred = [], []
        fold_counter = 0

        for fold_idx, held_out_group in enumerate(unique_groups, start=1):
            test_idx = np.where(groups == held_out_group)[0]
            train_idx = np.where(groups != held_out_group)[0]
            y_train = y[train_idx]

            if len(np.unique(y_train)) < 2:
                continue

            x_train = x[train_idx]
            x_test = x[test_idx]
            mean = x_train.mean(axis=(0, 1), keepdims=True)
            std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8

            preds = train_and_predict(
                (x_train - mean) / std, seq_lengths[train_idx], y_train,
                (x_test - mean) / std, seq_lengths[test_idx],
                64, 70, model_type
            )

            fold_counter += 1
            fold_y_true.extend(y[test_idx].tolist())
            fold_y_pred.extend(preds.tolist())

            for local_idx, pred_val, true_val in zip(test_idx, preds, y[test_idx]):
                predictions.append({
                    "model": model_type, "fold": fold_idx, "participant_id": groups[local_idx],
                    "y_true": int(true_val), "y_pred": int(pred_val),
                })

        if fold_counter > 0:
            results.append({
                "model": model_type, "n_folds": int(fold_counter), "n_samples": int(len(fold_y_true)),
                "accuracy": float(np.mean(np.asarray(fold_y_true) == np.asarray(fold_y_pred))),
                "balanced_accuracy": balanced_accuracy(fold_y_true, fold_y_pred),
                "f1": binary_f1(fold_y_true, fold_y_pred),
            })

    return pd.DataFrame(results), pd.DataFrame(predictions)


def save_plot(results_df, out_path):
    if results_df.empty: return
    plt.figure(figsize=(8, 4))
    plot_df = results_df.copy()
    plot_df["task_label"] = plot_df["task"].map(
        {"label_found_waldo": "Waldo găsit", "label_long_search": "Căutare lungă"}).fillna(plot_df["task"])
    plot_df["model_label"] = plot_df["model"].map({"mean_pool_mlp": "MLP", "bilstm_attention": "BiLSTM"}).fillna(
        plot_df["model"])
    plot_df["label"] = plot_df["task_label"] + " / " + plot_df["window_s"].astype(int).astype(str) + "s"
    plot_df = plot_df.sort_values(["window_s", "task_label", "model_label"])

    labels = plot_df["label"].drop_duplicates().to_list()
    x = np.arange(len(labels))
    width = 0.36

    for idx, model_label in enumerate(["MLP", "BiLSTM"]):
        model_df = plot_df[plot_df["model_label"] == model_label].set_index("label")
        values = [model_df.loc[label, "balanced_accuracy"] if label in model_df.index else np.nan for label in labels]
        plt.bar(x + (idx - 0.5) * width, values, width=width, label=model_label)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Balanced Accuracy")
    plt.xlabel("Fereastră de timp")
    plt.title("Performanța rețelelor MLP și BiLSTM pe Secvențe de Fixații")
    plt.legend()
    plt.tight_layout()
    save_figure(out_path, dpi=220)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saliency_maps = saliency_maps()
    all_results, all_predictions = [], []

    for window_s in [5.0, 10.0]:
        metadata_rows, padded_sequences, lengths = [], [], []

        for rec_id, intervals in WALDO_INTERVALS.items():
            fix_path = RECORDINGS_DIR / rec_id / "fixations_on_surface_Surface 1.csv"
            waldo_path = WALDO_DIR / f"waldo_fixations_{rec_id}.csv"

            if not fix_path.exists(): continue
            fix_df = pd.read_csv(fix_path)
            waldo_df = pd.read_csv(waldo_path) if waldo_path.exists() else pd.DataFrame()
            if fix_df.empty: continue

            add_relative_time_columns(fix_df, start_col="start timestamp [ns]", end_col="end timestamp [ns]",
                                      duration_col="duration [ms]")

            for level_name, start_s, end_s in intervals:
                l_fix = fix_df[
                    (fix_df["start_s"] >= start_s) & (fix_df["start_s"] <= min(end_s, start_s + window_s))].copy()
                if len(l_fix) < 3: continue

                xs, ys = l_fix["fixation x [normalized]"].to_numpy(dtype=float), l_fix[
                    "fixation y [normalized]"].to_numpy(dtype=float)
                durs = pd.to_numeric(l_fix["duration [ms]"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0

                dx, dy = np.diff(xs, prepend=xs[0]), np.diff(ys, prepend=ys[0])
                saccade_len = np.sqrt(dx ** 2 + dy ** 2)
                velocity = saccade_len / (durs + 1e-5)
                saccade_angle = np.arctan2(dy, dx) / np.pi

                seq = np.stack([
                    xs, ys, durs, saccade_len, saccade_angle, velocity,
                    np.diff(velocity, prepend=velocity[0]),
                    np.diff(saccade_angle, prepend=saccade_angle[0]),
                    saliency_values(xs, ys, saliency_maps.get(level_name))
                ], axis=1).astype(np.float32)

                seq_len = len(seq)
                seq = seq[:MAX_SEQ_LEN] if seq_len >= MAX_SEQ_LEN else np.vstack(
                    [seq, np.zeros((MAX_SEQ_LEN - seq_len, 9), dtype=np.float32)])

                padded_sequences.append(seq)
                lengths.append(min(seq_len, MAX_SEQ_LEN))
                metadata_rows.append({
                    "recording_id": rec_id, "participant_id": participant_id_from_recording(rec_id),
                    "level_name": level_name, "window_s": float(window_s),
                    "total_search_time_s": float(end_s - start_s),
                    "label_found_waldo": int(not waldo_df[waldo_df[
                                                              "waldo"] == level_name].empty) if not waldo_df.empty and "waldo" in waldo_df.columns else 0,
                })

        if not metadata_rows: continue
        meta_df = long_search_labels(pd.DataFrame(metadata_rows))
        meta_df.to_csv(OUTPUT_DIR / f"sequence_dataset_{int(window_s)}s.csv", index=False)

        seq_array, len_array = np.asarray(padded_sequences, dtype=np.float32), np.asarray(lengths, dtype=np.int64)

        for task in ["label_found_waldo", "label_long_search"]:
            result_df, pred_df = evaluate_sequence_task(meta_df, seq_array, len_array, task)
            if not result_df.empty:
                result_df["window_s"], result_df["task"] = window_s, task
                pred_df["window_s"], pred_df["task"] = window_s, task
                all_results.append(result_df)
                all_predictions.append(pred_df)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_df.to_csv(OUTPUT_DIR / "sequence_results.csv", index=False)
        if all_predictions:
            pd.concat(all_predictions, ignore_index=True).to_csv(OUTPUT_DIR / "sequence_predictions.csv", index=False)
        save_plot(results_df, OUTPUT_DIR / "sequence_balanced_accuracy.png")