import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from utils import (
    FEATURES_DIR,
    SEED,
    ROOT_DIR,
    SENSITIVITY_DIR,
    recordings as WALDO_INTERVALS,
    balanced_accuracy_multiclass,
    participant_id_from_recording,
)


OUTPUT_DIR = ROOT_DIR / "clasificare_participanti"


metrici = {
    "level_duration_s": "Durata",
    "ttff_waldo_s": "TTFF",
    "total_fixations": "Fixații",
    "fixation_rate_per_s": "Rată fixații",
    "scanpath_length_per_s": "Scanpath/s",
    "spatial_coverage_hull": "Acoperire",
    "fixation_grid_entropy_4x4": "Entropie 4x4",
    "transition_entropy_4x4": "Entropie tranziții",
    "gaze_entropy": "Entropie privire",
    "waldo_fixation_ratio": "Raport Waldo",
    "waldo_time_ratio": "Timp Waldo",
    "direct_waldo_ratio": "Hit direct",
    "mean_distance_to_target": "Distanță țintă",
    "peripheral_gaze_rate": "Hit periferic",
}


behaviour_groups = {
    "explorare_larga": [
        "scanpath_length_per_s",
        "spatial_coverage_hull",
        "fixation_grid_entropy_4x4",
        "transition_entropy_4x4",
        "gaze_entropy",
    ],
    "orientare_eficienta": [
        "level_duration_s",
        "ttff_waldo_s",
        "mean_distance_to_target",
    ],
    "focus_pe_tinta": [
        "waldo_fixation_ratio",
        "waldo_time_ratio",
        "direct_waldo_ratio",
        "peripheral_gaze_rate",
    ],
    "ritm_vizual": [
        "fixation_rate_per_s",
        "scanpath_length_per_s",
        "total_fixations",
    ],
}


labels_behaviour = {
    "explorare_larga": "Explorare extinsă",
    "orientare_eficienta": "Orientare eficientă",
    "focus_pe_tinta": "Focus pe personaj",
    "ritm_vizual": "Ritm vizual",
}


METRICI = {"level_duration_s", "ttff_waldo_s", "mean_distance_to_target"}


def save_figure(fig, path, **kwargs):
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **kwargs)
    fig.savefig(path.with_suffix(".pdf"), **{k: v for k, v in kwargs.items() if k != "dpi"})


def participant_order(extra_ids=None):
    ids = [participant_id_from_recording(recording_id) for recording_id in WALDO_INTERVALS]
    if extra_ids is not None:
        for participant_id in sorted(set(extra_ids)):
            if participant_id not in ids:
                ids.append(participant_id)
    return ids


def participant_labels(extra_ids=None):
    return {
        participant_id: f"Participant {idx + 1}"
        for idx, participant_id in enumerate(participant_order(extra_ids))
    }


def load_feat():
    features_path = FEATURES_DIR / "features_summary.csv"

    if features_path.exists():
        df = pd.read_csv(features_path)
        df["participant_id"] = df["recording_id"].astype(str).map(participant_id_from_recording)
    else:
        return pd.DataFrame()

    df = df.copy()
    if "participant_id" not in df.columns:
        df["participant_id"] = df["recording_id"].astype(str).map(participant_id_from_recording)

    df["participant_id"] = df["participant_id"].astype(str)
    df["level_name"] = df["level_name"].astype(str)

    for col in metrici:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ttff_waldo_s"] = df["ttff_waldo_s"].fillna(df["level_duration_s"])
    df["waldo_fixation_ratio"] = df["waldo_fixation_ratio"].fillna(0.0)
    df["waldo_time_ratio"] = df["waldo_time_ratio"].fillna(0.0)
    df["direct_waldo_ratio"] = df["direct_waldo_ratio"].fillna(0.0)
    df["peripheral_gaze_rate"] = df["peripheral_gaze_rate"].fillna(0.0)
    return df


def add_flags(df):
    flags_path = SENSITIVITY_DIR / "quality_exclusion_flags.csv"
    if not flags_path.exists():
        df["exclude_any_quality"] = False
        return df

    flags = pd.read_csv(flags_path)
    if "exclude_any_quality" not in flags.columns:
        df["exclude_any_quality"] = False
        return df

    flags = flags[["recording_id", "exclude_any_quality"]].copy()
    out = df.merge(flags, on="recording_id", how="left")
    out["exclude_any_quality"] = out["exclude_any_quality"].fillna(False).astype(bool)
    return out


def level_norm(df):
    out = df.copy()

    for metric in metrici:
        z_col = f"{metric}_level_z"
        out[z_col] = np.nan

        for _, group in out.groupby("level_name"):
            values = group[metric].astype(float)
            mediana = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            scale = q3 - q1

            if not np.isfinite(scale) or scale <= 1e-9:
                scale = values.std(ddof=0)
            if not np.isfinite(scale) or scale <= 1e-9:
                scale = 1.0

            out.loc[group.index, z_col] = ((values - mediana) / scale).clip(-3, 3)

    return out


def add_scores(df):
    out = df.copy()

    for nume_stil, cols in behaviour_groups.items():
        valori = []
        for metric in cols:
            serie = out[f"{metric}_level_z"]
            if nume_stil == "orientare_eficienta" or metric in METRICI:
                serie = -serie
            valori.append(serie)

        out[nume_stil] = pd.concat(valori, axis=1).mean(axis=1)

    return out


def participant_features(df):
    z_cols = [f"{metric}_level_z" for metric in metrici]
    style_cols = list(behaviour_groups)

    agg = {col: "mean" for col in z_cols + style_cols}
    agg.update({
        "recording_id": "count",
        "level_duration_s": "median",
        "ttff_waldo_s": "median",
        "waldo_fixation_ratio": "median",
        "gaze_entropy": "median",
    })
    if "age_years" in df.columns:
        agg["age_years"] = "first"
    if "age_group" in df.columns:
        agg["age_group"] = "first"

    stiluri = df.groupby("participant_id").agg(agg).reset_index()
    stiluri = stiluri.rename(columns={
        "recording_id": "levels_used",
        "level_duration_s": "median_duration_s",
        "ttff_waldo_s": "median_ttff_s",
        "waldo_fixation_ratio": "median_waldo_fixation_ratio",
        "gaze_entropy": "median_gaze_entropy",
    })

    labels = participant_labels(stiluri["participant_id"])
    stiluri["participant_label"] = stiluri["participant_id"].map(labels).fillna(stiluri["participant_id"])
    stiluri["stil_dominant"] = stiluri[style_cols].idxmax(axis=1)
    return stiluri.sort_values("participant_label")


def r2_score(y, x):
    y = y.astype(float)
    if len(y) == 0:
        return 0.0

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def design_matrix(df, columns):
    pieces = [pd.Series(1.0, index=df.index, name="intercept")]
    for col in columns:
        pieces.append(pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True, dtype=float))
    return pd.concat(pieces, axis=1).to_numpy(dtype=float)


def get_signal(df):
    rows = []

    for metric, label in metrici.items():
        clean = df[[metric, "level_name", "participant_id"]].dropna()
        if len(clean) < 8:
            continue

        y = clean[metric].to_numpy(dtype=float)
        level_x = design_matrix(clean, ["level_name"])
        full_x = design_matrix(clean, ["level_name", "participant_id"])
        level_r2 = r2_score(y, level_x)
        full_r2 = r2_score(y, full_x)

        rows.append({
            "metric": metric,
            "metric_label": label,
            "level_r2": level_r2,
            "level_plus_participant_r2": full_r2,
            "participant_incremental_r2": max(0.0, full_r2 - level_r2),
        })

    return pd.DataFrame(rows).sort_values("participant_incremental_r2", ascending=False)


def matrice_caracteristici(df, feature_cols):
    rows = df.dropna(subset=feature_cols + ["participant_id"]).copy()
    return (
        rows[feature_cols].to_numpy(dtype=float),
        rows["participant_id"].to_numpy(dtype=str),
        rows.index.to_list(),
    )


def centroid_classify(df, feature_cols, nume_metoda):
    participant_ids = participant_order(df["participant_id"])
    labels = participant_labels(participant_ids)
    pid_to_idx = {participant_id: idx for idx, participant_id in enumerate(participant_ids)}
    x, y, source_index = matrice_caracteristici(df, feature_cols)

    y_true = []
    y_pred = []
    predictii = []

    for row_idx in range(len(x)):
        true_pid = y[row_idx]
        train_mask = np.ones(len(x), dtype=bool)
        train_mask[row_idx] = False

        scoruri = {}
        for candidate_pid in participant_ids:
            candidate_mask = train_mask & (y == candidate_pid)
            if not candidate_mask.any():
                continue
            centroid = x[candidate_mask].mean(axis=0)
            scoruri[candidate_pid] = float(np.linalg.norm(x[row_idx] - centroid))

        if not scoruri or true_pid not in pid_to_idx:
            continue

        pred_pid = min(scoruri, key=scoruri.get)
        y_true.append(pid_to_idx[true_pid])
        y_pred.append(pid_to_idx[pred_pid])
        predictii.append({
            "source_row": source_index[row_idx],
            "participant_real_id": true_pid,
            "participant_prezis_id": pred_pid,
            "participant_real": labels.get(true_pid, true_pid),
            "participant_prezis": labels.get(pred_pid, pred_pid),
            "metoda": nume_metoda,
            "distanta_predictie": scoruri[pred_pid],
            "corect": int(true_pid == pred_pid),
        })

    n_classes = len(participant_ids)
    confuzie = pd.DataFrame(
        0,
        index=[labels[pid] for pid in participant_ids],
        columns=[labels[pid] for pid in participant_ids],
    )
    for true_idx, pred_idx in zip(y_true, y_pred):
        confuzie.iloc[true_idx, pred_idx] += 1

    rezultat = {
        "metoda": nume_metoda,
        "n_features": len(feature_cols),
        "n_participants": n_classes,
        "n_test_segments": len(y_true),
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if y_true else 0.0,
        "balanced_accuracy": balanced_accuracy_multiclass(y_true, y_pred, n_classes),
        "chance_balanced_accuracy": 1.0 / n_classes if n_classes else np.nan,
        "feature_columns": ";".join(feature_cols),
    }

    return rezultat, pd.DataFrame(predictii), confuzie


def calc_dist(df, feature_cols, stiluri):
    centers = stiluri.set_index("participant_id")[feature_cols].to_dict(orient="index")
    rows = []

    for _, row in df.dropna(subset=feature_cols + ["participant_id"]).iterrows():
        point = row[feature_cols].to_numpy(dtype=float)
        distante = {}

        for pid, center_row in centers.items():
            center = np.array([center_row[col] for col in feature_cols], dtype=float)
            distante[pid] = float(np.linalg.norm(point - center))

        own = distante.get(row["participant_id"], np.nan)
        others = [dist for pid, dist in distante.items() if pid != row["participant_id"]]
        rows.append({
            "participant_id": row["participant_id"],
            "level_name": row["level_name"],
            "distanta_stil_propriu": own,
            "distanta_cel_mai_apropiat_alt_stil": min(others) if others else np.nan,
            "stil_propriu_mai_apropiat": int(np.isfinite(own) and others and own < min(others)),
        })

    return pd.DataFrame(rows)


def top_clas(semnal_df):
    top_metrics = semnal_df["metric"].to_list()
    return {
        "toate_metricile": [f"{metric}_level_z" for metric in metrici],
        "top4_semnal_individual": [f"{metric}_level_z" for metric in top_metrics[:4]],
        "top5_semnal_individual": [f"{metric}_level_z" for metric in top_metrics[:5]],
        "top8_semnal_individual": [f"{metric}_level_z" for metric in top_metrics[:8]],
        "scoruri_de_stil": list(behaviour_groups),
        "explorare_larga": [f"{metric}_level_z" for metric in behaviour_groups["explorare_larga"]],
        "focus_pe_tinta": [
            f"{metric}_level_z"
            for metric in ["waldo_fixation_ratio", "waldo_time_ratio", "direct_waldo_ratio", "mean_distance_to_target", "ttff_waldo_s"]
        ],
        "ritm_vizual": [f"{metric}_level_z" for metric in ["fixation_rate_per_s", "scanpath_length_per_s", "total_fixations"]],
    }


def pca_2d(matrix):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    coords = centered @ components
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
    return coords


def save_heatmap(stiluri):
    z_cols = [f"{metric}_level_z" for metric in metrici]
    data = stiluri[z_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(15, 6.4))
    image = ax.imshow(data, cmap="RdBu_r", vmin=-1.8, vmax=1.8, aspect="auto")
    ax.set_xticks(range(len(z_cols)))
    ax.set_xticklabels(
        [metrici[col.replace("_level_z", "")] for col in z_cols],
        rotation=35,
        ha="right",
        fontsize=12,
    )
    ax.set_yticks(range(len(stiluri)))
    ax.set_yticklabels(stiluri["participant_label"], fontsize=12)
    ax.set_title("Tipare comportamentale normalizate per nivel", fontsize=14)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Abatere față de mediană")
    colorbar.ax.tick_params(labelsize=11)
    colorbar.set_label("Abatere față de mediană", fontsize=12)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "harta_stiluri_participanti.png", dpi=220)
    plt.close(fig)


def save_scores(stiluri):
    style_cols = list(behaviour_groups)
    x = np.arange(len(stiluri))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for idx, stil in enumerate(style_cols):
        ax.bar(x + (idx - 1.5) * width, stiluri[stil], width=width, label=labels_behaviour.get(stil, stil.replace("_", " ")))

    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stiluri["participant_label"], rotation=30, ha="right")
    ax.set_ylabel("Scor normalizat")
    ax.set_title("Distribuția tiparelor comportamentale")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "scoruri_stil_participanti.png", dpi=220)
    plt.close(fig)


def save_pca(stiluri):
    z_cols = [f"{metric}_level_z" for metric in metrici]
    coords = pca_2d(stiluri[z_cols].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8, 6))
    if "age_group" in stiluri.columns:
        age_groups = stiluri["age_group"]
    else:
        age_groups = pd.Series(["necunoscut"] * len(stiluri), index=stiluri.index)

    unique_groups = sorted(age_groups.fillna("necunoscut").unique())
    colors = dict(zip(unique_groups, plt.cm.Set2(np.linspace(0, 1, len(unique_groups)))))

    for idx, row in stiluri.reset_index(drop=True).iterrows():
        group = age_groups.iloc[idx] if pd.notna(age_groups.iloc[idx]) else "necunoscut"
        ax.scatter(coords[idx, 0], coords[idx, 1], s=90, color=colors[group], edgecolor="black")
        ax.text(coords[idx, 0], coords[idx, 1], " " + row["participant_label"].replace("Participant ", "P"), va="center")

    ax.axhline(0, color="#dddddd")
    ax.axvline(0, color="#dddddd")
    ax.set_xlabel("Componenta profil 1")
    ax.set_ylabel("Componenta profil 2")
    ax.set_title("Separarea profilurilor individuale în spațiul PCA")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "pca_stiluri_participanti.png", dpi=220)
    plt.close(fig)


def save_r2(semnal_df):
    plot_df = semnal_df.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["metric_label"], plot_df["participant_incremental_r2"], color="#4C78A8")
    ax.set_xlabel("Creșterea scorului R2")
    ax.set_title("Variația explicată de factorul individual")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "semnal_individual_r2.png", dpi=220)
    plt.close(fig)


def save_dist(distante):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        distante["distanta_stil_propriu"].dropna().to_numpy(),
        distante["distanta_cel_mai_apropiat_alt_stil"].dropna().to_numpy(),
    ]
    ax.boxplot(data, tick_labels=["Propriul profil", "Cel mai apropiat alt profil"])
    ax.set_ylabel("Distanță euclidiană pe caracteristici normalizate")
    ax.set_title("Separarea intra/inter-participant")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "separare_distante_clasificare.png", dpi=220)
    plt.close(fig)


def save_conf_matr(confuzie):
    normalizata = confuzie.div(confuzie.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    labels = [label.replace("Participant ", "P") for label in normalizata.index]

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    image = ax.imshow(normalizata.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(normalizata.columns)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(normalizata.index)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Participant prezis")
    ax.set_ylabel("Participant real")
    ax.set_title("Matrice de confuzie per participant")

    for i in range(normalizata.shape[0]):
        for j in range(normalizata.shape[1]):
            value = normalizata.iloc[i, j]
            if value > 0:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Proporție")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "matrice_confuzie_clasificare.png", dpi=220)
    plt.close(fig)
    return normalizata


def save_ans(df_all, df, stiluri, semnal_df, clasificare, distante):
    best = clasificare.sort_values("balanced_accuracy", ascending=False).iloc[0]
    rows = [
        {"indicator": "segmente_initiale", "valoare": len(df_all)},
        {"indicator": "segmente_folosite", "valoare": len(df)},
        {"indicator": "participanti", "valoare": df["participant_id"].nunique()},
        {"indicator": "segmente_fara_fixatii_eliminate", "valoare": int((df_all["total_fixations"].fillna(0) <= 0).sum())},
        {"indicator": "segmente_cu_flag_calitate_pastrate", "valoare": int(df["exclude_any_quality"].sum())},
        {"indicator": "cea_mai_buna_metoda", "valoare": best["metoda"]},
        {"indicator": "balanced_accuracy", "valoare": float(best["balanced_accuracy"])},
        {"indicator": "balanced_accuracy_sansa", "valoare": float(best["chance_balanced_accuracy"])},
        {"indicator": "stil_propriu_mai_apropiat_medie", "valoare": float(distante["stil_propriu_mai_apropiat"].mean())},
    ]

    for _, row in semnal_df.head(5).iterrows():
        rows.append({
            "indicator": f"semnal_individual_{row['metric']}",
            "valoare": float(row["participant_incremental_r2"]),
        })

    for _, row in stiluri.iterrows():
        rows.append({
            "indicator": f"stil_dominant_{row['participant_label'].replace(' ', '_').lower()}",
            "valoare": row["stil_dominant"],
        })

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "rezumat_clasificare.csv", index=False)


if __name__ == "__main__":
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_all = load_feat()
    if df_all.empty:
        print("Nu am gasit features_summary.csv.")
        exit(-1)

    df_all = add_flags(df_all)
    df_all["zero_fixations"] = df_all["total_fixations"].fillna(0) <= 0
    df_all["valid_for_classification"] = ~df_all["zero_fixations"]

    df = df_all[df_all["valid_for_classification"]].copy()
    df = level_norm(df)
    df = add_scores(df)

    stiluri = participant_features(df)
    semnal_df = get_signal(df)
    variante = top_clas(semnal_df)

    rezultate = []
    predictii = []
    matrici = {}

    for nume_metoda, feature_cols in variante.items():
        rezultat, pred_df, confuzie = centroid_classify(df, feature_cols, nume_metoda)
        rezultate.append(rezultat)
        predictii.append(pred_df)
        matrici[nume_metoda] = confuzie

    clasificare = pd.DataFrame(rezultate).sort_values("balanced_accuracy", ascending=False)
    predictii = pd.concat(predictii, ignore_index=True) if predictii else pd.DataFrame()

    best_method = clasificare.iloc[0]["metoda"]
    toate_z = [f"{metric}_level_z" for metric in metrici]
    distante = calc_dist(df, toate_z, stiluri)
    confuzie = matrici[best_method]
    confuzie_norm = save_conf_matr(confuzie)

    df.to_csv(OUTPUT_DIR / "segmente_clasificare.csv", index=False)
    stiluri.to_csv(OUTPUT_DIR / "stiluri_participanti.csv", index=False)
    semnal_df.to_csv(OUTPUT_DIR / "semnal_individual_metrici.csv", index=False)
    distante.to_csv(OUTPUT_DIR / "distante_clasificare.csv", index=False)
    clasificare.to_csv(OUTPUT_DIR / "rezultate_clasificare.csv", index=False)
    predictii.to_csv(OUTPUT_DIR / "predictii_clasificare.csv", index=False)
    confuzie.to_csv(OUTPUT_DIR / "matrice_confuzie_counts.csv")
    confuzie_norm.to_csv(OUTPUT_DIR / "matrice_confuzie_normalizata.csv")

    save_heatmap(stiluri)
    save_scores(stiluri)
    save_pca(stiluri)
    save_r2(semnal_df)
    save_dist(distante)
    save_ans(df_all, df, stiluri, semnal_df, clasificare, distante)

    print("Clasificarea participantilor a fost finalizata.")
    print(f"Output: {OUTPUT_DIR}")
