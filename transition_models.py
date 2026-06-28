import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


from utils import (
    SEED,
    RECORDINGS_DIR,
    TRANSITION_MODELS_OUTPUTS as OUTPUT_DIR,
    recordings as WALDO_INTERVALS,
    add_relative_time_columns,
    balanced_accuracy_multiclass,
    macro_f1,
    participant_id_from_recording,
    set_seed,
)

NUM_QUADRANTS = 4


def quadrant_id(x, y):
    if x < 0.5 and y < 0.5: return 0
    if x >= 0.5 and y < 0.5: return 1
    if x < 0.5 and y >= 0.5: return 2
    return 3


def one_hot_quad(q):
    arr = np.zeros(NUM_QUADRANTS, dtype=np.float32)
    arr[int(q)] = 1.0
    return arr


def predict_majority(y_train, x_test):
    return np.full(len(x_test), int(np.argmax(np.bincount(y_train.astype(int), minlength=NUM_QUADRANTS))), dtype=int)


def predict_markov(x_train, y_train, x_test):
    counts = np.ones((NUM_QUADRANTS, NUM_QUADRANTS), dtype=float)
    srcs = np.argmax(x_train[:, :NUM_QUADRANTS], axis=1)
    for src, dst in zip(srcs, y_train):
        counts[int(src), int(dst)] += 1.0

    probs = counts / counts.sum(axis=1, keepdims=True)
    test_srcs = np.argmax(x_test[:, :NUM_QUADRANTS], axis=1)
    return np.argmax(probs[test_srcs], axis=1).astype(int)


def predict_torch(x_train, y_train, x_test, hidden_dim=32, epochs=220, lr=5e-3):
    set_seed(SEED)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(hidden_dim, NUM_QUADRANTS),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return torch.argmax(model(torch.tensor(x_test, dtype=torch.float32)), dim=1).cpu().numpy().astype(int)


def save_plot(results_df, out_path):
    if results_df.empty: return
    try:
        plot_df = results_df.copy()
        plot_df["label"] = plot_df["model"] + " (" + plot_df["window_s"].astype(int).astype(str) + "s)"

        plt.figure(figsize=(9, 5))
        plt.bar(plot_df["label"], plot_df["balanced_accuracy"], color="#70ad47")
        plt.ylabel("Balanced Accuracy")
        plt.xlabel("Model / Fereastră Timp")
        plt.title("Predicția următoarei regiuni de privire (Tranziții)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except MemoryError:
        plt.close("all")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results, all_predictions = [], []

    for window_s in [5.0, 10.0]:
        rows = []
        for rec_id, intervals in WALDO_INTERVALS.items():
            fix_path = RECORDINGS_DIR / rec_id / "fixations_on_surface_Surface 1.csv"
            if not fix_path.exists(): continue
            fix_df = pd.read_csv(fix_path)
            if fix_df.empty: continue

            add_relative_time_columns(fix_df, start_col="start timestamp [ns]", end_col="end timestamp [ns]",
                                      duration_col="duration [ms]")

            for level_name, start_s, end_s in intervals:
                l_fix = fix_df[
                    (fix_df["start_s"] >= start_s) & (fix_df["start_s"] <= min(end_s, start_s + window_s))].copy()
                if len(l_fix) < 3: continue

                xs = l_fix["fixation x [normalized]"].to_numpy(dtype=float)
                ys = l_fix["fixation y [normalized]"].to_numpy(dtype=float)
                durs = pd.to_numeric(l_fix["duration [ms]"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / 1000.0
                quads = np.asarray([quadrant_id(x, y) for x, y in zip(xs, ys)], dtype=int)

                dx, dy = np.diff(xs), np.diff(ys)
                lengths = np.sqrt(dx ** 2 + dy ** 2)
                angles = np.arctan2(dy, dx) / np.pi

                for i in range(1, len(quads) - 1):
                    s_q, p_q, d_q = quads[i], quads[i - 1], quads[i + 1]
                    f_vec = np.concatenate([
                        one_hot_quad(s_q), one_hot_quad(p_q),
                        np.array([durs[i], lengths[i - 1] if i - 1 < len(lengths) else 0.0,
                                  angles[i - 1] if i - 1 < len(angles) else 0.0, xs[i], ys[i]], dtype=np.float32)
                    ])
                    row = {
                        "recording_id": rec_id, "participant_id": participant_id_from_recording(rec_id),
                        "level_name": level_name, "window_s": float(window_s),
                        "src_quad": int(s_q), "prev_quad": int(p_q), "dst_quad": int(d_q)
                    }
                    row.update({f"f_{j}": float(v) for j, v in enumerate(f_vec)})
                    rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty: continue
        df.to_csv(OUTPUT_DIR / f"transition_dataset_{int(window_s)}s.csv", index=False)

        f_cols = [c for c in df.columns if c.startswith("f_")]
        X = df[f_cols].to_numpy(dtype=float)
        y = df["dst_quad"].astype(int).to_numpy()
        groups = df["participant_id"].astype(str).to_numpy()

        if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2: continue
        unique_groups = list(dict.fromkeys(groups.tolist()))

        for model_name in ["majority", "markov_1step", "mlp_transition"]:
            f_y_true, f_y_pred = [], []
            f_count = 0

            for f_idx, h_out in enumerate(unique_groups, start=1):
                test_idx = np.where(groups == h_out)[0]
                train_idx = np.where(groups != h_out)[0]
                if len(train_idx) == 0 or len(test_idx) == 0: continue

                x_tr, x_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                mean = x_tr.mean(axis=0, keepdims=True)
                std = x_tr.std(axis=0, keepdims=True) + 1e-8

                x_tr_scaled = x_tr if model_name in ["majority", "markov_1step"] else (x_tr - mean) / std
                x_te_scaled = x_te if model_name in ["majority", "markov_1step"] else (x_te - mean) / std

                if model_name == "majority":
                    y_pred = predict_majority(y_tr, x_te_scaled)
                elif model_name == "markov_1step":
                    y_pred = predict_markov(x_tr_scaled, y_tr, x_te_scaled)
                else:
                    y_pred = predict_torch(x_tr_scaled, y_tr, x_te_scaled, epochs=220)

                f_count += 1
                f_y_true.extend(y_te.tolist())
                f_y_pred.extend(y_pred.tolist())

                for r_idx, p_val, t_val in zip(test_idx, y_pred, y_te):
                    all_predictions.append({
                        "model": model_name, "fold": f_idx, "participant_id": groups[r_idx],
                        "window_s": float(window_s), "y_true": int(t_val), "y_pred": int(p_val)
                    })

            if f_y_true:
                all_results.append({
                    "model": model_name, "window_s": float(window_s), "n_folds": int(f_count),
                    "n_samples": int(len(f_y_true)),
                    "accuracy": float(np.mean(np.asarray(f_y_true) == np.asarray(f_y_pred))),
                    "balanced_accuracy": balanced_accuracy_multiclass(f_y_true, f_y_pred,
                                                                              n_classes=NUM_QUADRANTS),
                    "macro_f1": macro_f1(f_y_true, f_y_pred, n_classes=NUM_QUADRANTS),
                })

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_DIR / "transition_results.csv", index=False)
        if all_predictions:
            pd.DataFrame(all_predictions).to_csv(OUTPUT_DIR / "transition_predictions.csv", index=False)
        save_plot(results_df, OUTPUT_DIR / "transition_balanced_accuracy.png")