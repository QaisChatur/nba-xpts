"""Probability-model evaluation: proper scoring rules, calibration, uncertainty, subgroups.

Ground truth is always the official make/miss. NBA xFG is treated as a competing model, never
as the target. All models in a comparison are scored on exactly the same rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pipeline.config import RANDOM_SEED

EPS = 1e-6


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def per_shot_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, float)
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bin_index(p, n_bins: int = 10, strategy: str = "quantile") -> np.ndarray:
    """Calibration bin per prediction; models with <= n_bins distinct values are binned by value."""
    p = np.asarray(p, float)
    values = np.unique(p)
    if len(values) <= n_bins:
        return np.searchsorted(values, p)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    return np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = bin_index(p, n_bins, strategy)
    ece = 0.0
    for b in np.unique(idx):
        m = idx == b
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b*logit(p). Ideal: b = 1, a = 0; b < 1 means over-confident."""
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    z = np.log(_clip(p) / (1 - _clip(p))).reshape(-1, 1)
    # lbfgs (scipy L-BFGS-B) segfaults with scipy 1.18.1 on Python 3.14/arm64.
    lr = LogisticRegression(C=np.inf, solver="newton-cholesky", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def prob_metrics(y, p) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    both = len(np.unique(y)) == 2
    slope, intercept = calibration_slope_intercept(y, p) if len(y) >= 30 else (float("nan"), float("nan"))
    return {
        "n": int(len(y)),
        "observed_fg": float(y.mean()),
        "mean_pred": float(p.mean()),
        "mean_signed_error": float((p - y).mean()),
        "mean_abs_error": float(np.abs(p - y).mean()),
        "log_loss": float(log_loss(y, _clip(p), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if both else float("nan"),
        "ece_quantile10": expected_calibration_error(y, p, 10, "quantile"),
        "ece_uniform10": expected_calibration_error(y, p, 10, "uniform"),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_pred_on_makes": float(p[y == 1].mean()) if (y == 1).any() else float("nan"),
        "mean_pred_on_misses": float(p[y == 0].mean()) if (y == 0).any() else float("nan"),
    }


def compare_models(df: pd.DataFrame, y_col: str, model_cols: dict[str, str]) -> pd.DataFrame:
    """Metrics table (rows = models) on identical rows; rows with any missing prediction are dropped up-front."""
    sub = df.dropna(subset=[y_col] + list(model_cols.values()))
    rows = []
    for name, col in model_cols.items():
        rows.append({"model": name, **prob_metrics(sub[y_col], sub[col])})
    return pd.DataFrame(rows).set_index("model")


def paired_bootstrap(df: pd.DataFrame, y_col: str, model_cols: dict[str, str], pairs: list[tuple[str, str]],
                     n_boot: int = 2000, seed: int = RANDOM_SEED, cluster_col: str | None = None) -> pd.DataFrame:
    """Percentile CIs for each model's log loss / Brier / AUC and for paired differences (first - second).

    Resamples shots (or clusters if `cluster_col` given). With few clusters, cluster bootstrap is not
    meaningful, so shot-level resampling is the default; it ignores within-game correlation and
    therefore understates uncertainty.
    """
    sub = df.dropna(subset=[y_col] + list(model_cols.values())).reset_index(drop=True)
    y = sub[y_col].to_numpy(int)
    preds = {k: sub[c].to_numpy(float) for k, c in model_cols.items()}
    ll = {k: per_shot_log_loss(y, p) for k, p in preds.items()}
    br = {k: (p - y) ** 2 for k, p in preds.items()}
    rng = np.random.default_rng(seed)
    if cluster_col:
        groups = sub.groupby(cluster_col).indices
        keys = list(groups)
    stats: dict[str, list] = {}
    for _ in range(n_boot):
        if cluster_col:
            pick = rng.choice(len(keys), len(keys), replace=True)
            idx = np.concatenate([groups[keys[i]] for i in pick])
        else:
            idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        auc_ok = len(np.unique(yb)) == 2
        cur = {}
        for k in preds:
            cur[(k, "log_loss")] = ll[k][idx].mean()
            cur[(k, "brier")] = br[k][idx].mean()
            cur[(k, "roc_auc")] = roc_auc_score(yb, preds[k][idx]) if auc_ok else np.nan
        for key, v in cur.items():
            stats.setdefault(f"{key[0]}|{key[1]}", []).append(v)
        for a, b in pairs:
            for m in ("log_loss", "brier", "roc_auc"):
                stats.setdefault(f"{a} - {b}|{m}", []).append(cur[(a, m)] - cur[(b, m)])
    rows = []
    for key, vals in stats.items():
        name, metric = key.split("|")
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        rows.append({"quantity": name, "metric": metric, "mean": v.mean(), "ci_low": np.quantile(v, 0.025),
                     "ci_high": np.quantile(v, 0.975), "share_below_zero": float((v < 0).mean())})
    return pd.DataFrame(rows)


def subgroup_table(df: pd.DataFrame, group_col: str, y_col: str, model_cols: dict[str, str], min_n: int = 1) -> pd.DataFrame:
    sub = df.dropna(subset=[y_col] + list(model_cols.values()))
    rows = []
    for g, d in sub.groupby(group_col, observed=True, dropna=False):
        if len(d) < min_n:
            continue
        r = {group_col: g, "n": len(d), "observed_fg": d[y_col].mean()}
        for name, col in model_cols.items():
            r[f"{name}_mean_pred"] = d[col].mean()
            r[f"{name}_log_loss"] = per_shot_log_loss(d[y_col], d[col]).mean()
            r[f"{name}_brier"] = ((d[col] - d[y_col]) ** 2).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def calibration_bins(y, p, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    idx = bin_index(p, n_bins, strategy)
    rows = []
    for b in np.unique(idx):
        m = idx == b
        k, n = y[m].sum(), m.sum()
        # Wilson 95% interval for the observed make rate in the bin.
        z = 1.96
        phat = k / n
        denom = 1 + z**2 / n
        centre = (phat + z**2 / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
        rows.append({"bin": int(b), "n": int(n), "mean_pred": p[m].mean(), "observed": phat,
                     "obs_ci_low": centre - half, "obs_ci_high": centre + half})
    return pd.DataFrame(rows)


def plot_calibration(df: pd.DataFrame, y_col: str, model_cols: dict[str, str], path, title: str, n_bins: int = 8) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df.dropna(subset=[y_col] + list(model_cols.values()))
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7, 8.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect calibration")
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    for (name, col), c in zip(model_cols.items(), colors):
        cb = calibration_bins(sub[y_col], sub[col], n_bins)
        ax.errorbar(cb["mean_pred"], cb["observed"], yerr=[cb["observed"] - cb["obs_ci_low"], cb["obs_ci_high"] - cb["observed"]],
                    marker="o", capsize=3, lw=1.5, c=c, label=f"{name}")
        axh.hist(sub[col], bins=30, range=(0, 1), histtype="step", lw=1.5, color=c, label=name)
    ax.set_ylabel("observed make rate (95% Wilson CI)")
    ax.set_title(f"{title}\n(equal-count bins, n={len(sub)} shots)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)
    axh.set_xlabel("predicted make probability")
    axh.set_ylabel("shots")
    axh.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def disagreement_analysis(df: pd.DataFrame, y_col: str, a_col: str, b_col: str, context_cols: list[str], top: int = 12) -> dict:
    sub = df.dropna(subset=[y_col, a_col, b_col]).copy()
    sub["diff"] = sub[a_col] - sub[b_col]
    sub["abs_diff"] = sub["diff"].abs()
    sub["ll_a"] = per_shot_log_loss(sub[y_col], sub[a_col])
    sub["ll_b"] = per_shot_log_loss(sub[y_col], sub[b_col])
    big = sub[sub["abs_diff"] >= 0.15]
    cols = context_cols + [y_col, a_col, b_col, "diff"]
    return {
        "pearson_r": float(sub[a_col].corr(sub[b_col])),
        "spearman_r": float(sub[a_col].corr(sub[b_col], method="spearman")),
        "mean_abs_diff": float(sub["abs_diff"].mean()),
        "share_abs_diff_ge_0.10": float((sub["abs_diff"] >= 0.10).mean()),
        "share_abs_diff_ge_0.15": float((sub["abs_diff"] >= 0.15).mean()),
        "large_disagreements": {
            "n": int(len(big)),
            "observed_fg": float(big[y_col].mean()) if len(big) else None,
            "mean_a": float(big[a_col].mean()) if len(big) else None,
            "mean_b": float(big[b_col].mean()) if len(big) else None,
            "log_loss_a": float(big["ll_a"].mean()) if len(big) else None,
            "log_loss_b": float(big["ll_b"].mean()) if len(big) else None,
        },
        "top_a_higher": sub.sort_values("diff", ascending=False).head(top)[cols],
        "top_b_higher": sub.sort_values("diff").head(top)[cols],
    }
