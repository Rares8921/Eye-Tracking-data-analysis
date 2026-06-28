import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import (
    CLASSICAL_MODELS_OUTPUTS as OUTPUT_DIR,
    SEED,
    RECORDINGS_DIR,
    waldo_coords as WALDO_BOXES,
    WALDO_DIR,
    recordings as WALDO_INTERVALS,
    add_relative_time_columns,
    long_search_labels,
    balanced_accuracy,
    binary_f1,
    cell_entropy,
    convex_hull_area,
    gaze_entropy,
    grid_cell_ids,
    transition_entropy,
    saliency_maps,
    participant_id_from_recording,
    saliency_values,
)

def save_figure(path, **kwargs):
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(path, **kwargs)
    plt.savefig(path.with_suffix(".pdf"), **{k: v for k, v in kwargs.items() if k != "dpi"})


def predict_dummy(y_train, x_test):
    majority_class = int(np.argmax(np.bincount(y_train.astype(int))))
    return np.full(len(x_test), majority_class, dtype=int)


def predict_centroid(x_train, y_train, x_test):
    centroids = {int(cls): x_train[y_train == cls].mean(axis=0) for cls in np.unique(y_train)}
    return np.asarray([min(centroids.keys(), key=lambda cls: np.linalg.norm(row - centroids[cls])) for row in x_test],
                      dtype=int)


def predict_torch(x_train, y_train, x_test, hidden_dim, epochs, lr):
    torch.manual_seed(SEED)
    input_dim = x_train.shape[1]

    if hidden_dim <= 0:
        model = nn.Linear(input_dim, 1)
    else:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1),
        )

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    pos_weight = torch.tensor([float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)],
                              dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(x_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return (torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32)).squeeze(1)) >= 0.5).cpu().numpy().astype(
            int)


def evaluate_task(df, feature_columns, label_column, groups):
    feature_df = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    labels = pd.to_numeric(df[label_column], errors="coerce")
    mask = labels.notna()
    X = feature_df.loc[mask].to_numpy(dtype=float)
    y = labels.loc[mask].astype(int).to_numpy()
    group_values = groups.loc[mask].to_numpy()

    if len(np.unique(y)) < 2 or len(np.unique(group_values)) < 2:
        return pd.DataFrame(), pd.DataFrame()

    results, predictions = [], []
    unique_groups = list(dict.fromkeys(group_values.tolist()))

    model_configs = {
        "dummy_majority": None,
        "nearest_centroid": None,
        "logreg_torch": {"hidden_dim": 0, "epochs": 220, "lr": 1e-2},
        "mlp_torch": {"hidden_dim": 32, "epochs": 260, "lr": 5e-3},
    }

    for model_name, kwargs in model_configs.items():
        fold_y_true, fold_y_pred = [], []
        fold_counter = 0

        for fold_idx, held_out_group in enumerate(unique_groups, start=1):
            test_idx = np.where(group_values == held_out_group)[0]
            train_idx = np.where(group_values != held_out_group)[0]
            y_train = y[train_idx]

            if len(np.unique(y_train)) < 2:
                continue

            x_train = X[train_idx]
            x_test = X[test_idx]
            mean = x_train.mean(axis=0, keepdims=True)
            std = x_train.std(axis=0, keepdims=True) + 1e-8

            x_train_scaled = x_train if model_name == "dummy_majority" else (x_train - mean) / std
            x_test_scaled = x_test if model_name == "dummy_majority" else (x_test - mean) / std

            if model_name == "dummy_majority":
                y_pred = predict_dummy(y_train, x_test_scaled)
            elif model_name == "nearest_centroid":
                y_pred = predict_centroid(x_train_scaled, y_train, x_test_scaled)
            else:
                y_pred = predict_torch(x_train_scaled, y_train, x_test_scaled, **kwargs)

            fold_counter += 1
            fold_y_true.extend(y[test_idx].tolist())
            fold_y_pred.extend(y_pred.tolist())

            for row_idx, pred_val, true_val in zip(test_idx, y_pred, y[test_idx]):
                predictions.append({
                    "model": model_name, "fold": fold_idx,
                    "participant_id": group_values[row_idx],
                    "y_true": int(true_val), "y_pred": int(pred_val),
                })

        if fold_y_true:
            results.append({
                "model": model_name, "n_folds": int(fold_counter), "n_samples": int(len(fold_y_true)),
                "accuracy": float(np.mean(np.asarray(fold_y_true) == np.asarray(fold_y_pred))),
                "balanced_accuracy": balanced_accuracy(fold_y_true, fold_y_pred),
                "f1": binary_f1(fold_y_true, fold_y_pred),
            })

    return pd.DataFrame(results), pd.DataFrame(predictions)


def save_benchmark_plot(results_df, out_path):
    if results_df.empty: return
    try:
        plot_df = results_df.copy()
        plot_df = plot_df[plot_df["model"] != "dummy_majority"]
        if plot_df.empty: return
        plot_df["window_label"] = plot_df["window_s"].map(lambda x: f"{int(x)}s")
        plot_df["task_label"] = plot_df["task"].map(
            {"label_found_waldo": "Predicția succesului", "label_long_search": "Predicția căutării lungi"}).fillna(
            plot_df["task"]) + " / " + plot_df["feature_set"].map(
            {"gaze_only": "doar gaze", "gaze_plus_target": "gaze + relația cu ținta"}).fillna(plot_df["feature_set"])

        model_labels = {"logreg_torch": "Regresie logistică", "mlp_torch": "MLP", "nearest_centroid": "Centroid"}
        tasks = [t for t in ["label_found_waldo", "label_long_search"] if t in set(plot_df["task"])]
        f_sets = [f for f in ["gaze_only", "gaze_plus_target"] if f in set(plot_df["feature_set"])]

        fig, axes = plt.subplots(len(tasks), len(f_sets), figsize=(10.5, 6.4), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)

        handles_by_label = {}
        for r_idx, task in enumerate(tasks):
            for c_idx, f_set in enumerate(f_sets):
                ax = axes[r_idx, c_idx]
                subset = plot_df[(plot_df["task"] == task) & (plot_df["feature_set"] == f_set)]
                for m_name, m_group in subset.groupby("model"):
                    m_group = m_group.sort_values("window_s")
                    line, = ax.plot(m_group["window_label"], m_group["balanced_accuracy"], marker="o", linewidth=1.8,
                                    label=model_labels.get(m_name, m_name))
                    handles_by_label[line.get_label()] = line
                ax.set_title(f"{task}\n{f_set}", fontsize=10)
                ax.set_ylim(0, 1)
                ax.grid(axis="y", alpha=0.25)
                if c_idx == 0: ax.set_ylabel("Balanced Accuracy")
                if r_idx == len(tasks) - 1: ax.set_xlabel("Fereastră de timp")

        fig.suptitle("Performanța modelelor pe primele ferestre de timp", fontsize=13)
        fig.legend(handles_by_label.values(), handles_by_label.keys(), loc="upper center", bbox_to_anchor=(0.5, 0.02),
                   ncol=len(handles_by_label), frameon=False)
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        save_figure(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    except MemoryError:
        plt.close("all")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saliency_maps = saliency_maps()
    all_results, all_predictions = [], []

    gaze_only_cols = [
        "early_fix_count", "early_fix_rate", "early_avg_fix_duration_s", "early_total_fix_duration_s",
        "early_saccade_length_avg", "early_saccade_length_median", "early_scanpath_length_total",
        "early_scanpath_length_per_s",
        "early_spatial_coverage_hull", "early_fixation_entropy_4x4", "early_transition_entropy_4x4",
        "early_unique_grid_cells_4x4",
        "early_mean_distance_to_target", "early_min_distance_to_target", "early_gaze_entropy", "early_gaze_count",
        "early_mean_saliency", "early_max_saliency"
    ]
    feature_sets = {"gaze_only": gaze_only_cols,
                    "gaze_plus_target": gaze_only_cols + ["early_waldo_hit_count", "early_direct_hit_count",
                                                          "early_peripheral_hit_count", "early_hit_any"]}
    tasks = ["label_found_waldo", "label_long_search"]

    for window_s in [3.0, 5.0, 10.0]:
        rows = []
        for rec_id, intervals in WALDO_INTERVALS.items():
            fix_path = RECORDINGS_DIR / rec_id / "fixations_on_surface_Surface 1.csv"
            gaze_path = RECORDINGS_DIR / rec_id / "gaze_positions_on_surface_Surface 1.csv"
            waldo_path = WALDO_DIR / f"waldo_fixations_{rec_id}.csv"

            if not fix_path.exists() or not gaze_path.exists(): continue

            fix_df = pd.read_csv(fix_path)
            gaze_df = pd.read_csv(gaze_path)
            waldo_df = pd.read_csv(waldo_path) if waldo_path.exists() else pd.DataFrame()

            if fix_df.empty or gaze_df.empty: continue

            add_relative_time_columns(fix_df, start_col="start timestamp [ns]", end_col="end timestamp [ns]",
                                      duration_col="duration [ms]")
            add_relative_time_columns(gaze_df, start_col="timestamp [ns]")

            for level_name, start_s, end_s in intervals:
                eff_win = float(max(1e-6, min(end_s, start_s + window_s) - start_s))
                early_end = start_s + eff_win

                l_fix = fix_df[(fix_df["start_s"] >= start_s) & (fix_df["start_s"] <= early_end)].copy()
                l_gaze = gaze_df[(gaze_df["start_s"] >= start_s) & (gaze_df["start_s"] <= early_end)].copy()

                l_waldo_all, l_waldo_early = pd.DataFrame(), pd.DataFrame()
                if not waldo_df.empty and "waldo" in waldo_df.columns:
                    l_waldo_all = waldo_df[waldo_df["waldo"] == level_name].copy()
                    l_waldo_early = l_waldo_all[
                        (l_waldo_all["timestamp_s"] >= start_s) & (l_waldo_all["timestamp_s"] <= early_end)]

                target_box = WALDO_BOXES[level_name]
                t_cx, t_cy = (target_box[0] + target_box[2]) / 2.0, (target_box[1] + target_box[3]) / 2.0

                if not l_fix.empty:
                    xs, ys = l_fix["fixation x [normalized]"].to_numpy(dtype=float), l_fix[
                        "fixation y [normalized]"].to_numpy(dtype=float)
                    durs = pd.to_numeric(l_fix["duration [ms]"], errors="coerce").fillna(0.0).to_numpy(
                        dtype=float) / 1000.0
                    saccades = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) if len(xs) > 1 else np.array([],
                                                                                                         dtype=float)
                    g_ids = grid_cell_ids(xs, ys)
                    s_vals = saliency_values(xs, ys, saliency_maps.get(level_name))
                    dists = np.sqrt((xs - t_cx) ** 2 + (ys - t_cy) ** 2)

                    metrics = {
                        "early_fix_count": len(xs), "early_fix_rate": len(xs) / eff_win,
                        "early_avg_fix_duration_s": float(durs.mean()) if len(durs) else 0.0,
                        "early_total_fix_duration_s": float(durs.sum()),
                        "early_saccade_length_avg": float(saccades.mean()) if len(saccades) else 0.0,
                        "early_saccade_length_median": float(np.median(saccades)) if len(saccades) else 0.0,
                        "early_scanpath_length_total": float(saccades.sum()) if len(saccades) else 0.0,
                        "early_scanpath_length_per_s": float(saccades.sum()) / eff_win if len(saccades) else 0.0,
                        "early_spatial_coverage_hull": convex_hull_area(xs, ys),
                        "early_fixation_entropy_4x4": cell_entropy(g_ids),
                        "early_transition_entropy_4x4": transition_entropy(g_ids),
                        "early_unique_grid_cells_4x4": len(np.unique(g_ids)),
                        "early_mean_distance_to_target": float(dists.mean()),
                        "early_min_distance_to_target": float(dists.min()),
                        "early_mean_saliency": float(s_vals.mean()) if len(s_vals) else 0.0,
                        "early_max_saliency": float(s_vals.max()) if len(s_vals) else 0.0
                    }
                else:
                    metrics = {k: 0.0 for k in gaze_only_cols if k not in ["early_gaze_entropy", "early_gaze_count"]}
                    metrics["early_fix_count"], metrics["early_unique_grid_cells_4x4"] = 0, 0

                if not l_gaze.empty:
                    gx, gy = l_gaze["gaze position on surface x [normalized]"].to_numpy(dtype=float), l_gaze[
                        "gaze position on surface y [normalized]"].to_numpy(dtype=float)
                    metrics.update(
                        {"early_gaze_entropy": gaze_entropy(gx, gy, bins=20), "early_gaze_count": len(gx)})
                else:
                    metrics.update({"early_gaze_entropy": 0.0, "early_gaze_count": 0})

                found_waldo = int(not l_waldo_all.empty)
                row = {
                    "recording_id": rec_id, "participant_id": participant_id_from_recording(rec_id),
                    "level_name": level_name,
                    "window_s": float(window_s), "effective_window_s": eff_win,
                    "total_search_time_s": float(end_s - start_s),
                    "label_found_waldo": found_waldo, "label_first_hit_direct": int(
                        found_waldo and l_waldo_all.sort_values("timestamp_s").iloc[0][
                            "type"] == "direct") if found_waldo else np.nan,
                    "early_waldo_hit_count": len(l_waldo_early), "early_direct_hit_count": int(
                        (l_waldo_early["type"] == "direct").sum()) if not l_waldo_early.empty else 0,
                    "early_peripheral_hit_count": int(
                        (l_waldo_early["type"] == "peripheral").sum()) if not l_waldo_early.empty else 0,
                    "early_hit_any": int(len(l_waldo_early) > 0)
                }
                row.update(metrics)
                rows.append(row)

        dataset = long_search_labels(pd.DataFrame(rows)) if rows else pd.DataFrame()
        if dataset.empty: continue
        dataset.to_csv(OUTPUT_DIR / f"early_window_dataset_{int(window_s)}s.csv", index=False)
        groups = dataset["participant_id"].astype(str)

        for task in tasks:
            for f_set_name, f_cols in feature_sets.items():
                res_df, preds_df = evaluate_task(dataset, f_cols, task, groups)
                if not res_df.empty:
                    res_df["window_s"], res_df["task"], res_df["feature_set"] = window_s, task, f_set_name
                    preds_df["window_s"], preds_df["task"], preds_df["feature_set"] = window_s, task, f_set_name
                    all_results.append(res_df)
                    all_predictions.append(preds_df)

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        results_df.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)
        predictions_df.to_csv(OUTPUT_DIR / "benchmark_predictions.csv", index=False)
        save_benchmark_plot(results_df, OUTPUT_DIR / "benchmark_balanced_accuracy.png")