import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2


import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from utils import (
    EDA_DIR,
    FEATURES_DIR,
    RECORDINGS_DIR,
    SALIENCY_DIR,
    recordings as WALDO_INTERVALS,
    participant_id_from_recording,
    add_relative_time_columns,
    fixation_timestamp_is_truncated,
    parse_neon_summary_csv,
)

FIG_DIR = Path(EDA_DIR) / "figures"

PARTICIPANT_LABELS = {
    participant_id_from_recording(rec_id): f"Participant {idx + 1}"
    for idx, rec_id in enumerate(WALDO_INTERVALS)
}

LABEL_MAP = {
    "annotated_duration_s": "Durată căutare (s)",
    "gaze_points_raw": "Număr puncte privire",
    "gaze_entropy_raw": "Entropie privire brută",
    "total_fixations": "Număr fixații",
    "saccade_length_avg": "Lungime medie sacadă",
    "scanpath_length_total": "Lungime totală traseu",
    "fixation_dispersion": "Dispersie fixații",
    "spatial_coverage_hull": "Acoperire spațială",
    "waldo_fixation_ratio": "Raport fixații pe țintă",
    "gaze_entropy": "Entropie privire",
    "transition_entropy_4x4": "Entropie tranziții 4x4",
    "ttff_waldo_s": "TTFF (s)",
    "TTFF_s": "TTFF (s)",
    "NSS_all_vs_saliency": "NSS global",
    "AUCJ_fix_saliency": "AUC-Judd",
}

CORRELATION_LABEL_MAP = {
    "annotated_duration_s": "Durată",
    "gaze_points_raw": "Puncte privire",
    "total_fixations": "Fixații",
    "saccade_length_avg": "Sacadă medie",
    "scanpath_length_total": "Traseu total",
    "fixation_dispersion": "Dispersie",
    "spatial_coverage_hull": "Acoperire",
    "waldo_fixation_ratio": "Raport țintă",
    "gaze_entropy": "Entropie privire",
    "transition_entropy_4x4": "Entropie tranziții",
    "ttff_waldo_s": "TTFF",
    "NSS_all_vs_saliency": "NSS",
}


def save_figure(path, **kwargs):
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(path, **kwargs)
    plt.savefig(path.with_suffix(".pdf"), **{k: v for k, v in kwargs.items() if k != "dpi"})


def compute_entropy(x_values, y_values, bins=30):
    if len(x_values) == 0:
        return 0.0
    hist, _, _ = np.histogram2d(x_values, y_values, bins=bins, range=[[0, 1], [0, 1]])
    prob = hist.ravel()
    total = prob.sum()
    if total <= 0:
        return 0.0
    prob = prob / total
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def save_plot_level_duration(levels_df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=levels_df, x="level_name", y="annotated_duration_s", color="#f4b183")
    sns.stripplot(data=levels_df, x="level_name", y="annotated_duration_s", color="#7f6000", alpha=0.7, size=5)
    plt.title("Distribuția timpului de căutare pe nivel")
    plt.xlabel("Nivel")
    plt.ylabel("Durată (s)")
    plt.tight_layout()
    save_figure(FIG_DIR / "level_duration_boxplot.png", dpi=220)
    plt.close()


def save_plot_gaze_points(levels_df):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=levels_df, x="level_name", y="gaze_points_raw", color="#9dc3e6")
    sns.stripplot(data=levels_df, x="level_name", y="gaze_points_raw", color="#1f4e79", alpha=0.7, size=5)
    plt.title("Distribuția punctelor de privire per nivel")
    plt.xlabel("Nivel")
    plt.ylabel("Volum puncte (Gaze)")
    plt.tight_layout()
    save_figure(FIG_DIR / "gaze_points_by_level.png", dpi=220)
    plt.close()


def save_plot_recording_quality(quality_df):
    plot_df = quality_df[[
        "participant_id",
        "fix_out_of_bounds_pct",
        "fix_surface_low_conf_pct",
        "gaze_out_of_bounds_pct",
    ]].melt(id_vars="participant_id", var_name="metric", value_name="value")

    plot_df["participant_label"] = plot_df["participant_id"].map(PARTICIPANT_LABELS).fillna(plot_df["participant_id"])
    metric_map = {
        "fix_out_of_bounds_pct": "Fixații în afara suprafeței (%)",
        "fix_surface_low_conf_pct": "Fixații cu încredere scăzută (%)",
        "gaze_out_of_bounds_pct": "Puncte privire în afara suprafeței (%)",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_map)

    plt.figure(figsize=(11, 5))
    ax = sns.barplot(data=plot_df, x="participant_label", y="value", hue="metric")
    ax.legend(title="Metrică")
    plt.title("Indicatori de calitate per participant")
    plt.xlabel("Participant")
    plt.ylabel("Procent")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_figure(FIG_DIR / "recording_quality_overview.png", dpi=220, bbox_inches="tight")
    plt.close()


def save_plot_ttff(levels_df):
    if "TTFF_s" not in levels_df.columns or levels_df["TTFF_s"].dropna().empty:
        return
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=levels_df, x="level_name", y="TTFF_s", color="#c6e0b4")
    sns.stripplot(data=levels_df, x="level_name", y="TTFF_s", color="#385723", alpha=0.7, size=5)
    plt.title("Timpul până la prima fixație pe personaj")
    plt.xlabel("Nivel")
    plt.ylabel("TTFF (s)")
    plt.tight_layout()
    save_figure(FIG_DIR / "ttff_by_level.png", dpi=220)
    plt.close()


def save_plot_saliency(levels_df):
    required = {"level_name", "NSS_all_vs_saliency", "AUCJ_fix_saliency", "IG_fix_vs_centerbias"}
    if not required.issubset(levels_df.columns):
        return

    level_order = ["waldo_1", "waldo_2", "waldo_3", "waldo_4"]
    plot_df = levels_df[["level_name", "NSS_all_vs_saliency", "AUCJ_fix_saliency", "IG_fix_vs_centerbias"]].melt(
        id_vars="level_name", var_name="metric", value_name="value"
    )
    plot_df["level_name"] = pd.Categorical(plot_df["level_name"], categories=level_order, ordered=True)
    plot_df = plot_df.sort_values("level_name")
    plot_df["metric"] = plot_df["metric"].map({
        "NSS_all_vs_saliency": "NSS global",
        "AUCJ_fix_saliency": "AUC-Judd",
        "IG_fix_vs_centerbias": "IG vs. center bias",
    })

    plt.figure(figsize=(9.0, 5.2))
    ax = sns.boxplot(
        data=plot_df,
        x="level_name",
        y="value",
        hue="metric",
        order=level_order,
        showfliers=False,
    )
    ax.set_ylim(-1.0, 1.35)
    ax.legend(title="Metrică", loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3, frameon=True, fontsize=10,
              title_fontsize=10)
    ax.tick_params(axis="both", labelsize=12)
    plt.xlabel("Nivel", fontsize=13)
    plt.ylabel("Valoare metrică", fontsize=13)
    plt.tight_layout()
    save_figure(FIG_DIR / "saliency_metrics_by_level.png", dpi=220)
    plt.close()


def save_plot_feature_correlations(levels_df):
    feature_columns = [
        "annotated_duration_s", "gaze_points_raw", "total_fixations", "saccade_length_avg",
        "scanpath_length_total", "fixation_dispersion", "spatial_coverage_hull",
        "waldo_fixation_ratio", "gaze_entropy", "transition_entropy_4x4",
        "ttff_waldo_s", "NSS_all_vs_saliency",
    ]
    available = [col for col in feature_columns if col in levels_df.columns]
    if len(available) < 3:
        return

    corr = levels_df[available].corr(method="spearman", numeric_only=True)
    corr = corr.rename(index=CORRELATION_LABEL_MAP, columns=CORRELATION_LABEL_MAP)

    plt.figure(figsize=(8.6, 7.2))
    ax = sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", square=True, annot_kws={"size": 8},
                     cbar_kws={"shrink": 0.85})
    ax.tick_params(axis="both", labelsize=8)
    plt.title("Matricea de corelație Spearman a metricilor", fontsize=11)
    plt.tight_layout()
    save_figure(FIG_DIR / "feature_correlation_heatmap.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    quality_rows = []
    raw_level_rows = []

    for recording_id in sorted(WALDO_INTERVALS):
        rec_dir = Path(RECORDINGS_DIR) / recording_id
        fix_path = rec_dir / "fixations_on_surface_Surface 1.csv"
        gaze_path = rec_dir / "gaze_positions_on_surface_Surface 1.csv"

        fix_df = pd.read_csv(fix_path) if fix_path.exists() else pd.DataFrame()
        gaze_df = pd.read_csv(gaze_path) if gaze_path.exists() else pd.DataFrame()

        fix_in_bounds = (fix_df["fixation x [normalized]"].between(0, 1) & fix_df["fixation y [normalized]"].between(0,
                                                                                                                     1)) if not fix_df.empty else pd.Series(
            dtype=bool)
        gaze_in_bounds = (gaze_df["gaze position on surface x [normalized]"].between(0, 1) & gaze_df[
            "gaze position on surface y [normalized]"].between(0, 1)) if not gaze_df.empty else pd.Series(dtype=bool)

        visible_frames, total_frames = parse_neon_summary_csv(rec_dir / "surface_visibility.csv")
        surface_gaze, total_gaze = parse_neon_summary_csv(rec_dir / "surface_gaze_distribution.csv")

        quality_rows.append({
            "recording_id": recording_id,
            "participant_id": participant_id_from_recording(recording_id),
            "total_fixations_raw": int(len(fix_df)),
            "total_gaze_points_raw": int(len(gaze_df)),
            "fix_out_of_bounds_pct": float((~fix_in_bounds).mean() * 100) if len(fix_df) else 0.0,
            "fix_surface_low_conf_pct": float(
                (fix_df["fixation detected on surface"] < 1).mean() * 100) if not fix_df.empty else 0.0,
            "gaze_out_of_bounds_pct": float((~gaze_in_bounds).mean() * 100) if len(gaze_df) else 0.0,
            "fix_timestamp_truncated": bool(fixation_timestamp_is_truncated(fix_path)),
            "surface_visible_frames": int(visible_frames),
            "surface_total_frames": int(total_frames),
            "surface_visibility_ratio": float(visible_frames / total_frames) if total_frames else 0.0,
            "surface_gaze_count": int(surface_gaze),
            "surface_total_gaze_count": int(total_gaze),
            "surface_gaze_ratio": float(surface_gaze / total_gaze) if total_gaze else 0.0,
        })

        if not gaze_df.empty:
            add_relative_time_columns(gaze_df, start_col="timestamp [ns]")
            for level_name, start_s, end_s in WALDO_INTERVALS.get(recording_id, []):
                level_df = gaze_df[(gaze_df["start_s"] >= start_s) & (gaze_df["start_s"] <= end_s)].copy()
                duration_s = end_s - start_s
                gaze_points = len(level_df)
                raw_level_rows.append({
                    "recording_id": recording_id,
                    "participant_id": participant_id_from_recording(recording_id),
                    "level_name": level_name,
                    "annotated_duration_s": duration_s,
                    "gaze_points_raw": int(gaze_points),
                    "gaze_points_per_s": float(gaze_points / duration_s) if duration_s else 0.0,
                    "gaze_in_bounds_ratio_raw": float(
                        gaze_in_bounds.loc[level_df.index].mean()) if gaze_points else np.nan,
                    "gaze_entropy_raw": compute_entropy(level_df["gaze position on surface x [normalized]"],
                                                        level_df["gaze position on surface y [normalized]"]),
                })

    quality_df = pd.DataFrame(quality_rows)
    raw_levels_df = pd.DataFrame(raw_level_rows)

    merged_levels_df = raw_levels_df.copy()
    features_path = Path(FEATURES_DIR) / "features_summary.csv"
    if features_path.exists():
        merged_levels_df = merged_levels_df.merge(pd.read_csv(features_path), on=["recording_id", "level_name"],
                                                  how="left")

    saliency_path = Path(SALIENCY_DIR) / "saliency_metrics_summary.csv"
    if saliency_path.exists():
        merged_levels_df = merged_levels_df.merge(pd.read_csv(saliency_path), on=["recording_id", "level_name"],
                                                  how="left")

    if "TTFF_s" not in merged_levels_df.columns and "ttff_waldo_s" in merged_levels_df.columns:
        merged_levels_df["TTFF_s"] = merged_levels_df["ttff_waldo_s"]

    consistency_rows = []
    metrics = ["annotated_duration_s", "ttff_waldo_s", "waldo_fixation_ratio", "gaze_entropy", "transition_entropy_4x4",
               "spatial_coverage_hull"]
    avail_metrics = [m for m in metrics if m in merged_levels_df.columns]

    for participant_id, group in merged_levels_df.groupby("participant_id"):
        row = {"participant_id": participant_id, "levels_available": int(len(group))}
        for metric in avail_metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            mean_val = float(values.mean()) if not values.empty else np.nan
            std_val = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean_val
            row[f"{metric}_std"] = std_val
            row[f"{metric}_cv"] = float(std_val / mean_val) if np.isfinite(mean_val) and abs(
                mean_val) > 1e-12 else np.nan
        consistency_rows.append(row)
    participant_consistency_df = pd.DataFrame(consistency_rows)

    failure_df = pd.DataFrame()
    if "annotated_duration_s" in merged_levels_df.columns:
        dur_thresh = merged_levels_df["annotated_duration_s"].median()
        low_hit = pd.Series(False, index=merged_levels_df.index)
        if "waldo_fixations" in merged_levels_df.columns:
            low_hit |= merged_levels_df["waldo_fixations"].fillna(0) <= 1
        if "waldo_fixation_ratio" in merged_levels_df.columns:
            low_hit |= merged_levels_df["waldo_fixation_ratio"].fillna(0) <= merged_levels_df[
                "waldo_fixation_ratio"].median()

        failure_df = merged_levels_df[(merged_levels_df["annotated_duration_s"] >= dur_thresh) & low_hit].copy()
        sort_cols = [c for c in ["annotated_duration_s", "ttff_waldo_s", "waldo_fixations"] if c in failure_df.columns]
        if sort_cols:
            failure_df = failure_df.sort_values(sort_cols, ascending=[False] + [True] * (len(sort_cols) - 1))

    corr_rows = []
    pairs = [
        ("annotated_duration_s", "ttff_waldo_s"), ("annotated_duration_s", "waldo_fixation_ratio"),
        ("annotated_duration_s", "gaze_entropy"), ("annotated_duration_s", "transition_entropy_4x4"),
        ("ttff_waldo_s", "NSS_all_vs_saliency"), ("waldo_fixation_ratio", "NSS_all_vs_saliency"),
        ("spatial_coverage_hull", "transition_entropy_4x4"),
    ]

    for left, right in pairs:
        if left not in merged_levels_df.columns or right not in merged_levels_df.columns:
            continue
        pair_df = merged_levels_df[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair_df) < 3:
            continue
        try:
            pearson_p = float(stats.pearsonr(pair_df[left], pair_df[right]).pvalue)
        except Exception:
            pearson_p = np.nan
        try:
            spearman_p = float(stats.spearmanr(pair_df[left], pair_df[right]).pvalue)
        except Exception:
            spearman_p = np.nan

        corr_rows.append({
            "metric_left": left, "metric_right": right, "n_rows": int(len(pair_df)),
            "pearson_r": float(pair_df[left].corr(pair_df[right], method="pearson")),
            "spearman_rho": float(pair_df[left].corr(pair_df[right], method="spearman")),
            "pearson_p": pearson_p, "spearman_p": spearman_p,
        })

    correlation_df = pd.DataFrame(corr_rows)
    if not correlation_df.empty:
        valid_p = correlation_df["spearman_p"].to_numpy(dtype=float)
        adjusted = np.full(len(valid_p), np.nan, dtype=float)
        finite_idx = np.where(np.isfinite(valid_p))[0]
        if len(finite_idx):
            m = len(finite_idx)
            order = finite_idx[np.argsort(valid_p[finite_idx])]
            running = 0.0
            for rank, idx in enumerate(order, start=1):
                value = min(1.0, valid_p[idx] * (m - rank + 1))
                running = max(running, value)
                adjusted[idx] = running
        correlation_df["spearman_p_holm"] = adjusted

    omnibus_rows, pairwise_rows = [], []
    for outcome, outcome_label in [("annotated_duration_s", "Durata căutării"), ("ttff_waldo_s", "TTFF")]:
        if outcome not in merged_levels_df.columns:
            continue
        df = merged_levels_df[["participant_id", "level_name", outcome]].copy()
        df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
        df = df.dropna()
        if len(df) < 8 or df["level_name"].nunique() < 2:
            continue

        trans_out, trans_lbl = outcome, "none"
        positive_mask = df[outcome] > 0
        df = df.loc[positive_mask].copy()
        if len(df) >= 8:
            trans_out = f"{outcome}_log"
            df[trans_out] = np.log(df[outcome])
            trans_lbl = "natural_log"

        mod_type, est, p_val = "unavailable", np.nan, np.nan
        if smf is not None and anova_lm is not None:
            try:
                full = smf.ols(f"{trans_out} ~ C(participant_id) + C(level_name)", data=df).fit()
                reduced = smf.ols(f"{trans_out} ~ C(participant_id)", data=df).fit()
                p_val = float(anova_lm(reduced, full)["Pr(>F)"].iloc[1])
                est = float(full.rsquared)
                mod_type = "participant_fixed_effects_anova"
            except Exception:
                pass

        omnibus_rows.append({
            "outcome": outcome, "outcome_label": outcome_label, "model_type": mod_type,
            "test": "level_effect", "effect_size_proxy": est, "p_value": p_val,
            "n_rows": int(len(df)), "n_participants": int(df["participant_id"].nunique()), "transform": trans_lbl,
        })

        pivot = df.pivot_table(index="participant_id", columns="level_name", values=outcome, aggfunc="mean")
        level_names = [lvl for lvl in ["waldo_1", "waldo_2", "waldo_3", "waldo_4"] if lvl in pivot.columns]
        local_rows = []
        for i, left in enumerate(level_names):
            for right in level_names[i + 1:]:
                pair = pivot[[left, right]].dropna()
                if len(pair) < 4: continue
                diff = pair[left] - pair[right]
                try:
                    pair_p = float(stats.wilcoxon(diff).pvalue)
                except Exception:
                    pair_p = np.nan
                local_rows.append({
                    "outcome": outcome, "outcome_label": outcome_label, "comparison": f"{left} vs {right}",
                    "median_difference": float(np.median(diff)), "n_pairs": int(len(pair)), "p_value": pair_p,
                })

        if local_rows:
            local_df = pd.DataFrame(local_rows)
            pvals = local_df["p_value"].to_numpy(dtype=float)
            adjusted = np.full(len(pvals), np.nan, dtype=float)
            finite_idx = np.where(np.isfinite(pvals))[0]
            if len(finite_idx):
                m = len(finite_idx)
                order = finite_idx[np.argsort(pvals[finite_idx])]
                running = 0.0
                for rank, idx in enumerate(order, start=1):
                    value = min(1.0, pvals[idx] * (m - rank + 1))
                    running = max(running, value)
                    adjusted[idx] = running
            local_df["p_value_holm"] = adjusted
            pairwise_rows.extend(local_df.to_dict("records"))

    level_inference_df = pd.DataFrame(omnibus_rows)
    level_pairwise_df = pd.DataFrame(pairwise_rows)

    quality_df.to_csv(Path(EDA_DIR) / "recording_quality_summary.csv", index=False)
    raw_levels_df.to_csv(Path(EDA_DIR) / "raw_level_summary.csv", index=False)
    merged_levels_df.to_csv(Path(EDA_DIR) / "merged_level_summary.csv", index=False)
    participant_consistency_df.to_csv(Path(EDA_DIR) / "participant_consistency_summary.csv", index=False)
    failure_df.to_csv(Path(EDA_DIR) / "failure_analysis_summary.csv", index=False)
    correlation_df.to_csv(Path(EDA_DIR) / "correlation_summary.csv", index=False)
    level_inference_df.to_csv(Path(EDA_DIR) / "level_inference_summary.csv", index=False)
    level_pairwise_df.to_csv(Path(EDA_DIR) / "level_pairwise_summary.csv", index=False)

    save_plot_level_duration(raw_levels_df)
    save_plot_gaze_points(raw_levels_df)
    save_plot_recording_quality(quality_df)
    save_plot_ttff(merged_levels_df)
    save_plot_saliency(merged_levels_df)
    save_plot_feature_correlations(merged_levels_df)
