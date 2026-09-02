# -*- coding: utf-8 -*-
"""
TIDE-ST v5.8 — source-profile 70:20:10 split BEFORE pseudo-spot generation
==========================================================================

This script replaces the old workflow that first generated all pseudo-spots and
then split pseudo-spots. The corrected order is:

    source snRNA profiles
        -> remove Unknown
        -> within each infection time, stratified 70:20:10 split by cell type
        -> fit ALL auxiliary-target definitions/scaling on TRAIN source profiles only
        -> apply fixed target definitions to validation/test source profiles
        -> independently generate train/validation/test pseudo-spots from mutually
           exclusive source-profile pools
        -> fit expression scaler and cell-type weights on TRAIN pseudo-spots only
        -> Optuna search on TRAIN + VALIDATION only
        -> checkpoint selection on VALIDATION only
        -> one-time final evaluation on held-out TEST pseudo-spots

Important outputs include source-profile split assignments and a leakage audit.
The test pseudo-spots are therefore composed exclusively of source profiles that
were never used to generate training/validation pseudo-spots.

Expected source files (already used in the TIDE-ST project):
    E:\\TIDEST_v0_inputs\\sc_0h_host_commongenes_v0.h5ad
    E:\\TIDEST_v0_inputs\\sc_12h_host_commongenes_v0.h5ad
    E:\\TIDEST_v0_inputs\\sc_24h_host_commongenes_v0.h5ad
    E:\\TIDEST_v0_inputs\\sc_48h_host_commongenes_v0.h5ad

Required obs columns:
    celltype
    infection_burden_raw_fraction

The pre-existing infection_burden/global_program_score/residual_score columns
are NOT used to fit targets in this corrected pipeline. They are rebuilt using
training source profiles only, in accordance with the manuscript.
"""

import os
import gc
import json
import math
import pickle
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import optuna

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 0. Paths and global settings
# =============================================================================
INPUT_DIR = r"E:\TIDEST_v0_inputs"
PSEUDO_DIR = r"E:\TIDEST_v5p8_source_split721_pseudospots"
OUT_DIR = r"E:\TIDEST_v5p8_source_split721_lambda_discrete_model"

os.makedirs(PSEUDO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

TIME_POINTS = ["0h", "12h", "24h", "48h"]
TIME_TO_INDEX = {"0h": 0, "12h": 1, "24h": 2, "48h": 3}
N_TIME = 4

SC_FILES = {
    "0h": os.path.join(INPUT_DIR, "sc_0h_host_commongenes_v0.h5ad"),
    "12h": os.path.join(INPUT_DIR, "sc_12h_host_commongenes_v0.h5ad"),
    "24h": os.path.join(INPUT_DIR, "sc_24h_host_commongenes_v0.h5ad"),
    "48h": os.path.join(INPUT_DIR, "sc_48h_host_commongenes_v0.h5ad"),
}

# -----------------------------------------------------------------------------
# Source-profile split: this is the 70:20:10 split described in the manuscript.
# -----------------------------------------------------------------------------
DATA_PARTITION_SEED = 1234
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.20
TEST_FRACTION = 0.10

# -----------------------------------------------------------------------------
# Pseudo-spot counts.
# These totals preserve the historical project totals (2000/2000/2000/1000),
# but NOW they are allocated to train/validation/test AFTER source-profile split.
# Therefore test pseudo-spot counts are 200/200/200/100 by default.
# -----------------------------------------------------------------------------
TOTAL_PSEUDOSPOTS_PER_TIME = {
    "0h": 2000,
    "12h": 2000,
    "24h": 2000,
    "48h": 1000,
}
PSEUDOSPOT_SEED = 24680
MIN_PROFILES_PER_SPOT = 5
MAX_PROFILES_PER_SPOT = 20
REMOVE_UNKNOWN = True
PRINT_EVERY = 100

# Save split-specific h5ad files with the newly fitted auxiliary targets.
# These are useful as leakage-free references for RCTD/Tangram/etc.
SAVE_SPLIT_H5AD = True

# -----------------------------------------------------------------------------
# Source metadata / auxiliary targets
# -----------------------------------------------------------------------------
CELLTYPE_COL = "celltype"
RAW_BURDEN_COL = "infection_burden_raw_fraction"
BURDEN_COL = "infection_burden"
GLOBAL_COL = "global_program_score"
RESID_MESO_COL = "residual_score__Mesophyll_v2"
RESID_XYLEM_COL = "residual_score__Xylem_v2"
FOCUS_CELLTYPES = ["Mesophyll", "Xylem"]

ROBUST_Q_LOW = 0.01
ROBUST_Q_HIGH = 0.99

# Global host-response program: preserve the v1 project criteria.
GLOBAL_MIN_CELLS_PER_GROUP = 50
GLOBAL_MIN_GENE_STD = 1e-8
GLOBAL_MIN_BURDEN_STD = 1e-8
GLOBAL_MIN_ABS_RHO = 0.08
GLOBAL_MIN_VALID_GROUPS = 2
GLOBAL_MIN_SUPPORT_RATIO = 0.40
GLOBAL_TOP_N_UP = 80
GLOBAL_TOP_N_DOWN = 80

# Cell-type-specific residual programs.
RESID_MIN_CELLS_PER_TIME = 80
RESID_MIN_GENE_STD = 1e-8
RESID_TOP_N_UP = 80
RESID_TOP_N_DOWN = 80

# Spearman computation chunk size. Larger values are faster but use more RAM.
SPEARMAN_GENE_CHUNK = 512

# -----------------------------------------------------------------------------
# Model / training settings
# -----------------------------------------------------------------------------
TRAINING_SEED = 1234
BATCH_SIZE = 64
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 12

RUN_HPARAM_SEARCH = True
N_TRIALS = 40
SEARCH_NUM_EPOCHS = 50
SEARCH_PATIENCE = 8
STUDY_NAME = "TIDEST_v5p8_source_split721_lambda_discrete_search"

# If you intentionally want a completely fresh Optuna search in the same OUT_DIR,
# set this True once. Default False allows interrupted searches to resume.
RESET_OPTUNA_STUDY = False

LAMBDA_PROP = 1.0
# Defaults are used only if RUN_HPARAM_SEARCH=False.
LAMBDA_BURDEN = 0.3
LAMBDA_GLOBAL = 0.2
LAMBDA_RESID = 0.3
LAMBDA_DECOR = 0.1
LAMBDA_SEARCH_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]

PROP_WEIGHT_POWER = 0.5
PROP_WEIGHT_EPS = 1e-6
PROP_WEIGHT_CLIP_MIN = 0.7
PROP_WEIGHT_CLIP_MAX = 2.0
TIME_PROP_WEIGHTS = {
    "0h": 1.0,
    "12h": 1.2,
    "24h": 1.0,
    "48h": 1.0,
}

# Preserve the previous scaling behavior to avoid changing another factor at the
# same time as the source-level split correction.
STANDARD_SCALER_WITH_MEAN = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 1. Reproducibility utilities
# =============================================================================
def log(msg: str):
    print(msg, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def robust_scale(x: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_high <= q_low:
        raise ValueError(f"Invalid robust scaling range: q_low={q_low}, q_high={q_high}")
    y = (x - q_low) / (q_high - q_low + 1e-8)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def ensure_csr(X):
    if sparse.issparse(X):
        return X.tocsr()
    return sparse.csr_matrix(np.asarray(X))


def rows_to_dense(X, row_idx: np.ndarray) -> np.ndarray:
    sub = X[row_idx]
    if sparse.issparse(sub):
        return sub.toarray().astype(np.float32)
    return np.asarray(sub, dtype=np.float32)


def mean_expression_rows(X, row_idx: np.ndarray) -> np.ndarray:
    sub = X[row_idx]
    if sparse.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub, dtype=np.float32).mean(axis=0).astype(np.float32)


def nanmean_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0 or np.all(np.isnan(x)):
        return np.nan
    return float(np.nanmean(x))


def split_integer_total(total: int, fractions: Tuple[float, float, float]) -> Dict[str, int]:
    """Largest-remainder allocation so counts sum exactly to total."""
    names = ["train", "validation", "test"]
    raw = np.asarray(fractions, dtype=float) * int(total)
    base = np.floor(raw).astype(int)
    remainder = int(total) - int(base.sum())
    order = np.argsort(-(raw - base))
    for i in range(remainder):
        base[order[i % len(base)]] += 1
    return {name: int(n) for name, n in zip(names, base)}


# =============================================================================
# 2. Load source snRNA profiles and perform 70:20:10 SOURCE split
# =============================================================================
def load_source_adatas() -> Dict[str, sc.AnnData]:
    adatas = {}
    reference_genes = None

    for t, fp in SC_FILES.items():
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Source h5ad not found: {fp}")

        ad = sc.read_h5ad(fp)

        required = [CELLTYPE_COL, RAW_BURDEN_COL]
        missing = [c for c in required if c not in ad.obs.columns]
        if missing:
            raise ValueError(
                f"{fp} is missing required source-level columns: {missing}. "
                f"This corrected pipeline requires {RAW_BURDEN_COL} so burden scaling can be fit on TRAIN only."
            )

        ad.obs[CELLTYPE_COL] = ad.obs[CELLTYPE_COL].astype(str)

        if REMOVE_UNKNOWN:
            keep = ad.obs[CELLTYPE_COL].values != "Unknown"
            ad = ad[keep].copy()

        # The *_host_commongenes_v0.h5ad files are expected to share identical
        # ordered host-gene features. Fail rather than silently reordering/dropping.
        genes = ad.var_names.astype(str).to_numpy()
        if reference_genes is None:
            reference_genes = genes.copy()
        elif not np.array_equal(reference_genes, genes):
            raise ValueError(
                f"Gene order mismatch at {t}. The four source h5ad files must have "
                "the identical common-host-gene order used by TIDE-ST."
            )

        ad.obs["source_time"] = t
        ad.obs["source_profile_id"] = [f"{t}::{x}" for x in ad.obs_names.astype(str)]
        adatas[t] = ad
        log(f"[Loaded source] {t}: n_profiles={ad.n_obs}, n_genes={ad.n_vars}")

    return adatas


def split_one_time_stratified(ad: sc.AnnData, seed: int) -> pd.Series:
    """
    Exact overall 70:20:10 split within one infection time point, stratified by
    annotated cell type. Fallback to per-celltype deterministic allocation only
    if sklearn stratification is impossible because a rare stratum is too small.
    """
    n = ad.n_obs
    idx = np.arange(n)
    y = ad.obs[CELLTYPE_COL].astype(str).to_numpy()

    temp_fraction = VAL_FRACTION + TEST_FRACTION
    test_within_temp = TEST_FRACTION / temp_fraction

    try:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=temp_fraction,
            random_state=seed,
            stratify=y,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_within_temp,
            random_state=seed,
            stratify=y[temp_idx],
        )
    except ValueError as e:
        log(f"[WARN] sklearn stratified split failed: {e}")
        log("[WARN] Falling back to deterministic per-celltype allocation.")
        rng = np.random.default_rng(seed)
        train_parts, val_parts, test_parts = [], [], []

        for ct in sorted(pd.unique(y)):
            ct_idx = idx[y == ct].copy()
            rng.shuffle(ct_idx)
            counts = split_integer_total(
                len(ct_idx), (TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
            )
            n_tr = counts["train"]
            n_va = counts["validation"]
            train_parts.append(ct_idx[:n_tr])
            val_parts.append(ct_idx[n_tr:n_tr + n_va])
            test_parts.append(ct_idx[n_tr + n_va:])

        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
        test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)

    split = np.full(n, "", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "validation"
    split[test_idx] = "test"

    if np.any(split == ""):
        raise RuntimeError("Source split failed: some profiles were not assigned.")

    # Strong overlap/coverage checks.
    sets = {
        "train": set(np.where(split == "train")[0].tolist()),
        "validation": set(np.where(split == "validation")[0].tolist()),
        "test": set(np.where(split == "test")[0].tolist()),
    }
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("Source-profile split overlap detected.")
    if len(sets["train"] | sets["validation"] | sets["test"]) != n:
        raise RuntimeError("Source-profile split does not cover all profiles.")

    return pd.Series(split, index=ad.obs_names, name="source_split")


def assign_source_splits(adatas: Dict[str, sc.AnnData]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(TIME_POINTS):
        ad = adatas[t]
        split = split_one_time_stratified(ad, DATA_PARTITION_SEED + i)
        ad.obs["source_split"] = split.loc[ad.obs_names].values

        tmp = ad.obs[["source_profile_id", "source_time", CELLTYPE_COL, "source_split"]].copy()
        tmp["obs_name"] = ad.obs_names.astype(str)
        rows.append(tmp.reset_index(drop=True))

        log(f"\n[Source split: {t}]")
        log(str(ad.obs.groupby(["source_split", CELLTYPE_COL], observed=False).size()))
        log(str(ad.obs["source_split"].value_counts()))

    assignment = pd.concat(rows, ignore_index=True)
    assignment.to_csv(os.path.join(PSEUDO_DIR, "source_profile_split_7_2_1.csv"), index=False)

    summary = (
        assignment.groupby(["source_time", "source_split", CELLTYPE_COL], observed=False)
        .size().reset_index(name="n_profiles")
    )
    summary.to_csv(os.path.join(PSEUDO_DIR, "source_profile_split_7_2_1_summary.csv"), index=False)

    # Audit globally unique source IDs and zero cross-split overlap.
    if assignment["source_profile_id"].duplicated().any():
        dups = assignment.loc[assignment["source_profile_id"].duplicated(), "source_profile_id"].head().tolist()
        raise RuntimeError(f"Duplicate source_profile_id detected: {dups}")

    split_sets = {
        s: set(assignment.loc[assignment["source_split"] == s, "source_profile_id"].astype(str))
        for s in ["train", "validation", "test"]
    }
    overlap_tv = split_sets["train"] & split_sets["validation"]
    overlap_tt = split_sets["train"] & split_sets["test"]
    overlap_vt = split_sets["validation"] & split_sets["test"]
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError("Cross-split source-profile leakage detected before pseudo-spot generation.")

    audit = {
        "data_partition_seed": DATA_PARTITION_SEED,
        "fractions": {"train": TRAIN_FRACTION, "validation": VAL_FRACTION, "test": TEST_FRACTION},
        "n_source_profiles": int(len(assignment)),
        "n_train": int((assignment["source_split"] == "train").sum()),
        "n_validation": int((assignment["source_split"] == "validation").sum()),
        "n_test": int((assignment["source_split"] == "test").sum()),
        "overlap_train_validation": len(overlap_tv),
        "overlap_train_test": len(overlap_tt),
        "overlap_validation_test": len(overlap_vt),
        "status": "PASS",
    }
    with open(os.path.join(PSEUDO_DIR, "source_profile_leakage_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    log("\n[SOURCE-LEVEL LEAKAGE AUDIT] PASS: train/validation/test source profiles are mutually exclusive.")
    return assignment


# =============================================================================
# 3. Fit auxiliary-target definitions on TRAIN source profiles ONLY
# =============================================================================
def spearman_corr_columns(X: np.ndarray, y: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    """Chunked Spearman rho for each X column against y, with average tie ranks."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    if X.shape[0] != len(y):
        raise ValueError("X/y length mismatch in spearman_corr_columns")

    n, p = X.shape
    out = np.full(p, np.nan, dtype=np.float32)
    if n < 3 or np.std(y) < 1e-8:
        return out

    ry = rankdata(y, method="average").astype(np.float64)
    ry -= ry.mean()
    y_norm = np.sqrt(np.sum(ry ** 2))
    if y_norm < 1e-12:
        return out

    for start in range(0, p, chunk_size):
        end = min(p, start + chunk_size)
        block = X[:, start:end]
        raw_std = np.std(block, axis=0)
        valid = raw_std >= 1e-8
        if not np.any(valid):
            continue

        ranks = rankdata(block, axis=0, method="average").astype(np.float64)
        ranks -= ranks.mean(axis=0, keepdims=True)
        denom = np.sqrt(np.sum(ranks ** 2, axis=0)) * y_norm
        corr = np.sum(ranks * ry[:, None], axis=0) / np.maximum(denom, 1e-12)
        corr[(denom < 1e-12) | (~valid)] = np.nan
        out[start:end] = corr.astype(np.float32)

    return out


def fit_train_only_burden(adatas: Dict[str, sc.AnnData]) -> Dict[str, float]:
    train_logvals = []
    cached = {}

    for t, ad in adatas.items():
        raw = ad.obs[RAW_BURDEN_COL].to_numpy(dtype=np.float32)
        if np.any(~np.isfinite(raw)):
            raise ValueError(f"Non-finite values in {RAW_BURDEN_COL} at {t}")
        if np.any(raw < 0):
            raise ValueError(f"Negative values in {RAW_BURDEN_COL} at {t}")

        logvals = np.log1p(raw).astype(np.float32)
        cached[t] = logvals
        is_train = ad.obs["source_split"].values == "train"
        train_logvals.append(logvals[is_train])

    train_concat = np.concatenate(train_logvals)
    q01 = float(np.quantile(train_concat, ROBUST_Q_LOW))
    q99 = float(np.quantile(train_concat, ROBUST_Q_HIGH))

    for t, ad in adatas.items():
        ad.obs[BURDEN_COL] = robust_scale(cached[t], q01, q99)

    info = {"q01": q01, "q99": q99, "fit_split": "train_source_profiles_only"}
    log(f"[Burden train-only scaling] q01={q01:.10f}, q99={q99:.10f}")
    return info


def fit_global_program_train_only(adatas: Dict[str, sc.AnnData]) -> Dict:
    genes = adatas[TIME_POINTS[0]].var_names.astype(str).to_numpy()
    group_names = []
    group_corrs = []

    for t, ad in adatas.items():
        obs = ad.obs
        train_mask = obs["source_split"].values == "train"
        celltypes = obs[CELLTYPE_COL].astype(str).to_numpy()
        burden = obs[BURDEN_COL].to_numpy(dtype=np.float32)

        for ct in sorted(pd.unique(celltypes[train_mask])):
            idx = np.where(train_mask & (celltypes == ct))[0]
            if len(idx) < GLOBAL_MIN_CELLS_PER_GROUP:
                continue
            y = burden[idx]
            if np.std(y) < GLOBAL_MIN_BURDEN_STD:
                continue

            log(f"[Global program] {t} x {ct}: n_train={len(idx)}")
            Xg = rows_to_dense(ad.X, idx)
            rho = spearman_corr_columns(Xg, y, SPEARMAN_GENE_CHUNK)
            group_names.append(f"{t}__{ct}")
            group_corrs.append(rho)
            del Xg
            gc.collect()

    if not group_corrs:
        raise RuntimeError("No valid train time x celltype groups for global program construction.")

    R = np.vstack(group_corrs)  # groups x genes
    valid = np.isfinite(R)
    n_valid = valid.sum(axis=0)
    mean_rho = np.nanmean(R, axis=0)
    median_rho = np.nanmedian(R, axis=0)
    abs_mean_rho = np.nanmean(np.abs(R), axis=0)
    pos_support = np.nansum(R >= GLOBAL_MIN_ABS_RHO, axis=0)
    neg_support = np.nansum(R <= -GLOBAL_MIN_ABS_RHO, axis=0)
    pos_ratio = np.divide(pos_support, n_valid, out=np.zeros_like(pos_support, dtype=float), where=n_valid > 0)
    neg_ratio = np.divide(neg_support, n_valid, out=np.zeros_like(neg_support, dtype=float), where=n_valid > 0)

    summary = pd.DataFrame({
        "gene": genes,
        "n_valid_groups": n_valid,
        "mean_rho": mean_rho,
        "median_rho": median_rho,
        "pos_support": pos_support,
        "neg_support": neg_support,
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
        "abs_mean_rho": abs_mean_rho,
    })
    summary = summary[summary["n_valid_groups"] >= GLOBAL_MIN_VALID_GROUPS].copy()

    up = summary[
        (summary["pos_ratio"] >= GLOBAL_MIN_SUPPORT_RATIO) &
        (summary["mean_rho"] >= GLOBAL_MIN_ABS_RHO / 2)
    ].sort_values(
        ["pos_ratio", "mean_rho", "abs_mean_rho", "n_valid_groups"],
        ascending=[False, False, False, False]
    )

    down = summary[
        (summary["neg_ratio"] >= GLOBAL_MIN_SUPPORT_RATIO) &
        (summary["mean_rho"] <= -GLOBAL_MIN_ABS_RHO / 2)
    ].sort_values(
        ["neg_ratio", "mean_rho", "abs_mean_rho", "n_valid_groups"],
        ascending=[False, True, False, False]
    )

    if len(up) < GLOBAL_TOP_N_UP or len(down) < GLOBAL_TOP_N_DOWN:
        raise RuntimeError(
            f"Training-only global program yielded only {len(up)} eligible up and {len(down)} eligible down genes; "
            f"need {GLOBAL_TOP_N_UP}/{GLOBAL_TOP_N_DOWN}. Do not silently relax criteria; inspect the training split."
        )

    up_genes = up["gene"].head(GLOBAL_TOP_N_UP).tolist()
    down_genes = down["gene"].head(GLOBAL_TOP_N_DOWN).tolist()
    summary.to_csv(os.path.join(PSEUDO_DIR, "global_program_train_only_gene_summary.csv"), index=False)

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    up_idx = np.asarray([gene_to_idx[g] for g in up_genes], dtype=int)
    down_idx = np.asarray([gene_to_idx[g] for g in down_genes], dtype=int)

    raw_by_time = {}
    train_raw = []
    for t, ad in adatas.items():
        # Only 160 columns need to be densified.
        Xu = ad.X[:, up_idx]
        Xd = ad.X[:, down_idx]
        if sparse.issparse(Xu):
            up_score = np.asarray(Xu.mean(axis=1)).ravel()
            down_score = np.asarray(Xd.mean(axis=1)).ravel()
        else:
            up_score = np.asarray(Xu).mean(axis=1)
            down_score = np.asarray(Xd).mean(axis=1)
        raw = (up_score - down_score).astype(np.float32)
        raw_by_time[t] = raw
        train_mask = ad.obs["source_split"].values == "train"
        train_raw.append(raw[train_mask])

    train_raw_all = np.concatenate(train_raw)
    q01 = float(np.quantile(train_raw_all, ROBUST_Q_LOW))
    q99 = float(np.quantile(train_raw_all, ROBUST_Q_HIGH))

    for t, ad in adatas.items():
        ad.obs["global_program_score_raw_trainfit"] = raw_by_time[t]
        ad.obs[GLOBAL_COL] = robust_scale(raw_by_time[t], q01, q99)

    info = {
        "fit_split": "train_source_profiles_only",
        "n_groups_used": len(group_names),
        "groups_used": group_names,
        "global_up_genes": up_genes,
        "global_down_genes": down_genes,
        "q01": q01,
        "q99": q99,
        "params": {
            "min_cells_per_group": GLOBAL_MIN_CELLS_PER_GROUP,
            "min_abs_rho": GLOBAL_MIN_ABS_RHO,
            "min_valid_groups": GLOBAL_MIN_VALID_GROUPS,
            "min_support_ratio": GLOBAL_MIN_SUPPORT_RATIO,
            "top_n_up": GLOBAL_TOP_N_UP,
            "top_n_down": GLOBAL_TOP_N_DOWN,
        },
    }
    log(f"[Global program train-only] selected {len(up_genes)} up / {len(down_genes)} down genes")
    return info


def fit_one_residual_program_train_only(
    adatas: Dict[str, sc.AnnData],
    celltype_name: str,
    output_col: str,
) -> Dict:
    genes = adatas[TIME_POINTS[0]].var_names.astype(str).to_numpy()
    time_corrs = []
    valid_times = []

    for t, ad in adatas.items():
        obs = ad.obs
        mask = (
            (obs["source_split"].values == "train") &
            (obs[CELLTYPE_COL].astype(str).values == celltype_name)
        )
        idx = np.where(mask)[0]
        if len(idx) < RESID_MIN_CELLS_PER_TIME:
            continue

        y = obs[BURDEN_COL].to_numpy(dtype=np.float32)[idx]
        if np.std(y) < 1e-8:
            continue

        log(f"[Residual {celltype_name}] {t}: n_train={len(idx)}")
        Xg = rows_to_dense(ad.X, idx)
        rho = spearman_corr_columns(Xg, y, SPEARMAN_GENE_CHUNK)
        time_corrs.append(rho)
        valid_times.append(t)
        del Xg
        gc.collect()

    if not time_corrs:
        raise RuntimeError(f"No valid training time points for residual program: {celltype_name}")

    R = np.vstack(time_corrs)
    mean_rho = np.nanmean(R, axis=0)
    median_rho = np.nanmedian(R, axis=0)
    n_times = np.sum(np.isfinite(R), axis=0)

    summary = pd.DataFrame({
        "gene": genes,
        "n_times": n_times,
        "mean_rho": mean_rho,
        "median_rho": median_rho,
    })
    summary = summary[summary["n_times"] > 0].copy()

    up = summary[summary["mean_rho"] > 0].sort_values(
        ["mean_rho", "median_rho"], ascending=[False, False]
    )
    down = summary[summary["mean_rho"] < 0].sort_values(
        ["mean_rho", "median_rho"], ascending=[True, True]
    )

    if len(up) < RESID_TOP_N_UP or len(down) < RESID_TOP_N_DOWN:
        raise RuntimeError(
            f"Training-only residual program {celltype_name} has insufficient positive/negative genes: "
            f"up={len(up)}, down={len(down)}."
        )

    up_genes = up["gene"].head(RESID_TOP_N_UP).tolist()
    down_genes = down["gene"].head(RESID_TOP_N_DOWN).tolist()
    summary.to_csv(
        os.path.join(PSEUDO_DIR, f"residual_{celltype_name}_train_only_gene_summary.csv"),
        index=False
    )

    gene_to_idx = {g: i for i, g in enumerate(genes)}
    up_idx = np.asarray([gene_to_idx[g] for g in up_genes], dtype=int)
    down_idx = np.asarray([gene_to_idx[g] for g in down_genes], dtype=int)

    # Compute raw program for this cell type in ALL splits, but fit regression on TRAIN only.
    records = []
    for t, ad in adatas.items():
        obs = ad.obs
        ct_mask = obs[CELLTYPE_COL].astype(str).values == celltype_name
        idx = np.where(ct_mask)[0]
        if len(idx) == 0:
            continue

        Xu = ad.X[idx][:, up_idx]
        Xd = ad.X[idx][:, down_idx]
        if sparse.issparse(Xu):
            up_score = np.asarray(Xu.mean(axis=1)).ravel()
            down_score = np.asarray(Xd.mean(axis=1)).ravel()
        else:
            up_score = np.asarray(Xu).mean(axis=1)
            down_score = np.asarray(Xd).mean(axis=1)
        program_raw = (up_score - down_score).astype(np.float32)

        burden = obs[BURDEN_COL].to_numpy(dtype=np.float32)[idx]
        global_score = obs[GLOBAL_COL].to_numpy(dtype=np.float32)[idx]
        split = obs["source_split"].astype(str).to_numpy()[idx]
        source_ids = obs["source_profile_id"].astype(str).to_numpy()[idx]

        for k in range(len(idx)):
            records.append((t, int(idx[k]), source_ids[k], split[k], program_raw[k], burden[k], global_score[k]))

    rec = pd.DataFrame(
        records,
        columns=["time", "row_idx", "source_profile_id", "source_split", "program_raw", "burden", "global_score"]
    )

    train_rec = rec[rec["source_split"] == "train"].copy()
    predictors_train = train_rec[["burden", "global_score"]].to_numpy(dtype=np.float32)
    target_train = train_rec["program_raw"].to_numpy(dtype=np.float32)

    reg = LinearRegression()
    reg.fit(predictors_train, target_train)

    predictors_all = rec[["burden", "global_score"]].to_numpy(dtype=np.float32)
    pred_all = reg.predict(predictors_all).astype(np.float32)
    rec["residual_raw"] = rec["program_raw"].to_numpy(dtype=np.float32) - pred_all

    train_resid_raw = rec.loc[rec["source_split"] == "train", "residual_raw"].to_numpy(dtype=np.float32)
    q01 = float(np.quantile(train_resid_raw, ROBUST_Q_LOW))
    q99 = float(np.quantile(train_resid_raw, ROBUST_Q_HIGH))
    rec["residual_scaled"] = robust_scale(rec["residual_raw"].to_numpy(dtype=np.float32), q01, q99)

    # Default NaN in all non-focus profiles.
    for ad in adatas.values():
        ad.obs[output_col] = np.nan

    for t, df_t in rec.groupby("time"):
        ad = adatas[t]
        values = ad.obs[output_col].to_numpy(dtype=np.float32)
        values[df_t["row_idx"].to_numpy(dtype=int)] = df_t["residual_scaled"].to_numpy(dtype=np.float32)
        ad.obs[output_col] = values

    rec.to_csv(os.path.join(PSEUDO_DIR, f"residual_{celltype_name}_trainfit_profile_table.csv"), index=False)

    info = {
        "celltype": celltype_name,
        "fit_split": "train_source_profiles_only",
        "valid_times": valid_times,
        "up_genes": up_genes,
        "down_genes": down_genes,
        "regression": {
            "coef_burden": float(reg.coef_[0]),
            "coef_global": float(reg.coef_[1]),
            "intercept": float(reg.intercept_),
        },
        "residual_q01": q01,
        "residual_q99": q99,
        "n_train_focus_profiles": int(len(train_rec)),
    }
    log(f"[Residual train-only] {celltype_name}: {len(up_genes)} up / {len(down_genes)} down")
    return info


def fit_all_auxiliary_targets(adatas: Dict[str, sc.AnnData]) -> Dict:
    burden_info = fit_train_only_burden(adatas)
    global_info = fit_global_program_train_only(adatas)
    meso_info = fit_one_residual_program_train_only(adatas, "Mesophyll", RESID_MESO_COL)
    xylem_info = fit_one_residual_program_train_only(adatas, "Xylem", RESID_XYLEM_COL)

    target_info = {
        "burden": burden_info,
        "global_program": global_info,
        "residual_programs": {
            "Mesophyll": meso_info,
            "Xylem": xylem_info,
        },
    }
    with open(os.path.join(PSEUDO_DIR, "train_only_auxiliary_target_definition.json"), "w", encoding="utf-8") as f:
        json.dump(target_info, f, indent=2, ensure_ascii=False)
    return target_info


# =============================================================================
# 4. Generate pseudo-spots independently from each SOURCE split
# =============================================================================
def sample_num_profiles(rng: np.random.Generator) -> int:
    return int(rng.integers(MIN_PROFILES_PER_SPOT, MAX_PROFILES_PER_SPOT + 1))


def sample_num_celltypes(rng: np.random.Generator, n_profiles: int, max_available: int) -> int:
    upper = min(n_profiles, max_available, 6)
    if upper < 1:
        raise RuntimeError("No available cell types for pseudo-spot generation.")

    # Preserve the historical TIDE-ST mixture-complexity sampling pattern.
    p = rng.random()
    if p < 0.30:
        low, high = 1, min(2, upper)
    elif p < 0.75:
        low, high = 2, min(4, upper)
    else:
        low, high = 4, min(6, upper)
    low = min(low, upper)
    high = max(low, high)
    return int(rng.integers(low, high + 1))


def choose_celltypes(valid_celltypes, valid_counts, k, rng):
    probs = valid_counts.astype(float) / valid_counts.sum()
    return rng.choice(valid_celltypes, size=min(k, len(valid_celltypes)), replace=False, p=probs)


def allocate_type_counts(k: int, n_profiles: int, rng: np.random.Generator) -> np.ndarray:
    if k > n_profiles:
        raise ValueError("k > n_profiles")
    counts = np.ones(k, dtype=int)
    remaining = n_profiles - k
    if remaining > 0:
        extra_props = rng.dirichlet(np.ones(k, dtype=np.float32))
        extra_counts = np.floor(extra_props * remaining).astype(int)
        counts += extra_counts
        leftover = remaining - int(extra_counts.sum())
        if leftover > 0:
            add_idx = rng.choice(np.arange(k), size=leftover, replace=True)
            for j in add_idx:
                counts[j] += 1
    if counts.sum() != n_profiles or np.any(counts < 1):
        raise RuntimeError("Pseudo-spot cell-type allocation failed.")
    return counts


def composition_from_labels(labels: List[str], all_celltypes: List[str]) -> np.ndarray:
    vc = pd.Series(labels).value_counts()
    vec = np.asarray([vc.get(ct, 0) for ct in all_celltypes], dtype=np.float32)
    return vec / max(float(vec.sum()), 1.0)


def generate_pseudospots_for_pool(
    ad: sc.AnnData,
    time_str: str,
    split_name: str,
    n_spots: int,
    all_celltypes: List[str],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, pd.DataFrame, set]:
    pool_mask = ad.obs["source_split"].astype(str).values == split_name
    pool_idx_global = np.where(pool_mask)[0]
    if len(pool_idx_global) == 0:
        raise RuntimeError(f"No source profiles in {time_str}/{split_name}")

    obs_pool = ad.obs.iloc[pool_idx_global].copy()
    labels_pool = obs_pool[CELLTYPE_COL].astype(str).to_numpy()
    celltype_counts = pd.Series(labels_pool).value_counts().sort_index()
    valid_celltypes = celltype_counts.index.to_numpy()
    valid_counts = celltype_counts.to_numpy(dtype=np.float64)

    local_by_ct = {ct: np.where(labels_pool == ct)[0] for ct in valid_celltypes}

    burden_pool = obs_pool[BURDEN_COL].to_numpy(dtype=np.float32)
    global_pool = obs_pool[GLOBAL_COL].to_numpy(dtype=np.float32)
    resid_meso_pool = obs_pool[RESID_MESO_COL].to_numpy(dtype=np.float32)
    resid_xylem_pool = obs_pool[RESID_XYLEM_COL].to_numpy(dtype=np.float32)
    source_ids_pool = obs_pool["source_profile_id"].astype(str).to_numpy()

    X_pool = ad.X[pool_idx_global]
    n_genes = ad.n_vars
    X_spots = np.zeros((n_spots, n_genes), dtype=np.float32)
    rows = []
    used_source_ids = set()

    for i in range(n_spots):
        n_profiles = sample_num_profiles(rng)
        n_ct = sample_num_celltypes(rng, n_profiles, len(valid_celltypes))
        selected_types = choose_celltypes(valid_celltypes, valid_counts, n_ct, rng)
        type_counts = allocate_type_counts(len(selected_types), n_profiles, rng)

        chosen_local = []
        chosen_labels = []
        for ct, need in zip(selected_types, type_counts):
            pool_local = local_by_ct[ct]
            # Reuse across different pseudo-spots is allowed within a split; cross-split
            # reuse is impossible because pools are source-disjoint.
            replace = len(pool_local) < int(need)
            idx_ct = rng.choice(pool_local, size=int(need), replace=replace)
            chosen_local.extend(idx_ct.tolist())
            chosen_labels.extend([str(ct)] * len(idx_ct))

        chosen_local = np.asarray(chosen_local, dtype=int)
        X_spots[i, :] = mean_expression_rows(X_pool, chosen_local)

        source_ids = source_ids_pool[chosen_local].astype(str).tolist()
        used_source_ids.update(source_ids)
        comp = composition_from_labels(chosen_labels, all_celltypes)

        row = {
            "spot_id": f"{split_name}_{time_str}_pseudo_{i}",
            "split": split_name,
            "time_str": time_str,
            "n_cells": int(len(chosen_local)),
            "n_profiles": int(len(chosen_local)),
            "n_celltypes": int(len(set(chosen_labels))),
            BURDEN_COL: float(np.mean(burden_pool[chosen_local])),
            GLOBAL_COL: float(np.mean(global_pool[chosen_local])),
            RESID_MESO_COL: nanmean_safe(resid_meso_pool[chosen_local]),
            RESID_XYLEM_COL: nanmean_safe(resid_xylem_pool[chosen_local]),
            # Audit trail only. Not used as a model feature.
            "source_profile_ids": "|".join(source_ids),
        }
        for ct, p in zip(all_celltypes, comp):
            row[f"prop__{ct}"] = float(p)
        rows.append(row)

        if (i + 1) % PRINT_EVERY == 0 or i + 1 == n_spots:
            log(f"[Pseudo] {split_name}/{time_str}: {i+1}/{n_spots}")

    return X_spots, pd.DataFrame(rows), used_source_ids


def save_split_h5ads(adatas: Dict[str, sc.AnnData]):
    if not SAVE_SPLIT_H5AD:
        return
    root = os.path.join(PSEUDO_DIR, "reference_splits")
    for split_name in ["train", "validation", "test"]:
        split_dir = os.path.join(root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for t, ad in adatas.items():
            mask = ad.obs["source_split"].astype(str).values == split_name
            ad_sub = ad[mask].copy()
            fp = os.path.join(split_dir, f"sc_{t}_{split_name}_host_commongenes_v0.h5ad")
            ad_sub.write(fp)
            log(f"[Saved source split h5ad] {fp} | shape={ad_sub.shape}")
            del ad_sub
            gc.collect()


def generate_all_split_pseudospots(adatas: Dict[str, sc.AnnData]) -> Dict[str, Dict[str, Tuple[np.ndarray, pd.DataFrame]]]:
    all_celltypes = sorted(set().union(*[
        set(ad.obs[CELLTYPE_COL].astype(str).unique()) for ad in adatas.values()
    ]))
    pd.Series(all_celltypes).to_csv(os.path.join(PSEUDO_DIR, "celltypes_v2_residual.csv"), index=False, header=False)

    counts_per_time = {
        t: split_integer_total(
            TOTAL_PSEUDOSPOTS_PER_TIME[t],
            (TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION)
        ) for t in TIME_POINTS
    }
    with open(os.path.join(PSEUDO_DIR, "pseudospot_counts_by_split.json"), "w", encoding="utf-8") as f:
        json.dump(counts_per_time, f, indent=2, ensure_ascii=False)

    out = {"train": {}, "validation": {}, "test": {}}
    used_by_split = {"train": set(), "validation": set(), "test": set()}

    for split_i, split_name in enumerate(["train", "validation", "test"]):
        split_dir = os.path.join(PSEUDO_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for time_i, t in enumerate(TIME_POINTS):
            n_spots = counts_per_time[t][split_name]
            rng = np.random.default_rng(PSEUDOSPOT_SEED + split_i * 1000 + time_i * 100)
            X, meta, used_ids = generate_pseudospots_for_pool(
                adatas[t], t, split_name, n_spots, all_celltypes, rng
            )
            np.save(os.path.join(split_dir, f"X_pseudospots_{t}.npy"), X)
            meta.to_csv(os.path.join(split_dir, f"meta_pseudospots_{t}.csv"), index=False)
            out[split_name][t] = (X, meta)
            used_by_split[split_name].update(used_ids)

    # Strong post-generation audit using actual source IDs used in pseudo-spots.
    tv = used_by_split["train"] & used_by_split["validation"]
    tt = used_by_split["train"] & used_by_split["test"]
    vt = used_by_split["validation"] & used_by_split["test"]
    audit = {
        "used_source_profiles_train": len(used_by_split["train"]),
        "used_source_profiles_validation": len(used_by_split["validation"]),
        "used_source_profiles_test": len(used_by_split["test"]),
        "overlap_train_validation": len(tv),
        "overlap_train_test": len(tt),
        "overlap_validation_test": len(vt),
        "status": "PASS" if not (tv or tt or vt) else "FAIL",
    }
    with open(os.path.join(PSEUDO_DIR, "pseudospot_source_overlap_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    if tv or tt or vt:
        raise RuntimeError("Pseudo-spot source-profile leakage detected.")
    log("\n[PSEUDO-SPOT SOURCE OVERLAP AUDIT] PASS: zero cross-split source-profile overlap.")

    # Backward-friendly benchmark folder containing ONLY the held-out test set.
    benchmark_test_dir = os.path.join(PSEUDO_DIR, "benchmark_test")
    os.makedirs(benchmark_test_dir, exist_ok=True)
    for t in TIME_POINTS:
        src_x = os.path.join(PSEUDO_DIR, "test", f"X_pseudospots_{t}.npy")
        src_m = os.path.join(PSEUDO_DIR, "test", f"meta_pseudospots_{t}.csv")
        shutil.copy2(src_x, os.path.join(benchmark_test_dir, f"X_pseudospots_{t}.npy"))
        shutil.copy2(src_m, os.path.join(benchmark_test_dir, f"meta_pseudospots_{t}.csv"))
    shutil.copy2(
        os.path.join(PSEUDO_DIR, "celltypes_v2_residual.csv"),
        os.path.join(benchmark_test_dir, "celltypes_v2_residual.csv")
    )

    return out


# =============================================================================
# 5. Build benchmark scenario assignments from HELD-OUT TEST pseudo-spots
# =============================================================================
def build_test_scenario_table(pseudospots) -> None:
    test_meta = pd.concat(
        [pseudospots["test"][t][1].copy() for t in TIME_POINTS],
        ignore_index=True
    )

    # Scenario B
    test_meta["scenario_B"] = np.select(
        [test_meta["n_celltypes"] <= 2, test_meta["n_celltypes"].between(3, 4), test_meta["n_celltypes"] >= 5],
        ["B1_simple", "B2_moderate", "B3_complex"],
        default="NA"
    )

    # Global tertiles for C and D. Rank-first avoids qcut failure from ties while
    # preserving the ordering implied by tertiles.
    test_meta["scenario_C"] = pd.qcut(
        test_meta["n_cells"].rank(method="first"),
        q=3,
        labels=["C1_small", "C2_medium", "C3_large"]
    ).astype(str)
    test_meta["scenario_D"] = pd.qcut(
        test_meta[BURDEN_COL].rank(method="first"),
        q=3,
        labels=["D1_low_infection", "D2_medium_infection", "D3_high_infection"]
    ).astype(str)

    test_meta.to_csv(os.path.join(PSEUDO_DIR, "test_scenario_assignments.csv"), index=False)

    rows = []
    for scenario_col in ["scenario_B", "scenario_C", "scenario_D"]:
        tmp = (
            test_meta.groupby([scenario_col, "time_str"], observed=False)
            .size().reset_index(name="n")
            .rename(columns={scenario_col: "group"})
        )
        tmp["scenario"] = scenario_col[-1]
        rows.append(tmp)
    counts = pd.concat(rows, ignore_index=True)
    counts.to_csv(os.path.join(PSEUDO_DIR, "test_scenario_counts_long.csv"), index=False)

    # Candidate Table S1 layout.
    pivot = counts.pivot_table(index=["scenario", "group"], columns="time_str", values="n", fill_value=0)
    for t in TIME_POINTS:
        if t not in pivot.columns:
            pivot[t] = 0
    pivot = pivot[TIME_POINTS].reset_index()
    pivot.to_csv(os.path.join(PSEUDO_DIR, "Table_S1_candidate_test_counts.csv"), index=False)
    log("[Saved] Table_S1_candidate_test_counts.csv from the true held-out TEST pseudo-spots")


# =============================================================================
# 6. TIDE-ST model / losses / metrics
# =============================================================================
class TIDESTDataset(Dataset):
    def __init__(self, X, time_idx, y_prop, y_burden, y_global, y_resid_meso, y_resid_xylem):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32)).float()
        self.time_idx = torch.from_numpy(np.asarray(time_idx, dtype=np.int64)).long()
        self.y_prop = torch.from_numpy(np.asarray(y_prop, dtype=np.float32)).float()
        self.y_burden = torch.from_numpy(np.asarray(y_burden, dtype=np.float32)).float()
        self.y_global = torch.from_numpy(np.asarray(y_global, dtype=np.float32)).float()
        self.y_resid_meso = torch.from_numpy(np.asarray(y_resid_meso, dtype=np.float32)).float()
        self.y_resid_xylem = torch.from_numpy(np.asarray(y_resid_xylem, dtype=np.float32)).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            self.X[idx], self.time_idx[idx], self.y_prop[idx], self.y_burden[idx],
            self.y_global[idx], self.y_resid_meso[idx], self.y_resid_xylem[idx]
        )


class TIDESTModel(nn.Module):
    def __init__(self, input_dim: int, n_celltypes: int, n_time: int = 4, time_hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.expr_encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(n_time, 16), nn.ReLU(),
            nn.Linear(16, time_hidden_dim), nn.ReLU(),
        )
        fused_dim = 64 + time_hidden_dim
        self.prop_head = nn.Linear(fused_dim, n_celltypes)
        self.burden_head = nn.Linear(fused_dim, 1)
        self.global_head = nn.Linear(fused_dim, 1)
        self.resid_meso_head = nn.Linear(fused_dim, 1)
        self.resid_xylem_head = nn.Linear(fused_dim, 1)
        self.n_time = n_time

    def forward(self, x, time_idx):
        h_expr = self.expr_encoder(x)
        onehot = torch.nn.functional.one_hot(time_idx, num_classes=self.n_time).float()
        h_time = self.time_encoder(onehot)
        h = torch.cat([h_expr, h_time], dim=1)
        return (
            torch.softmax(self.prop_head(h), dim=1),
            torch.sigmoid(self.burden_head(h)),
            torch.sigmoid(self.global_head(h)),
            torch.sigmoid(self.resid_meso_head(h)),
            torch.sigmoid(self.resid_xylem_head(h)),
        )


def masked_mse_loss(pred, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return (pred * 0.0).sum()
    return torch.mean((pred[mask] - target[mask]) ** 2)


class WeightedPropMSELoss(nn.Module):
    def __init__(self, celltype_weights: torch.Tensor, time_weights: torch.Tensor = None):
        super().__init__()
        self.register_buffer("celltype_weights", celltype_weights.view(1, -1))
        if time_weights is not None:
            self.register_buffer("time_weights", time_weights.view(-1))
        else:
            self.time_weights = None

    def forward(self, pred, target, time_idx=None):
        loss = (pred - target) ** 2 * self.celltype_weights
        if self.time_weights is not None and time_idx is not None:
            loss = loss * self.time_weights[time_idx].view(-1, 1)
        return loss.mean()


def _standardize_for_corr(z):
    z = z.view(z.size(0), -1)
    return (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)


def selective_decorrelation_loss(pred_burden, pred_global, pred_resid_meso, pred_resid_xylem):
    zb = _standardize_for_corr(pred_burden)
    zg = _standardize_for_corr(pred_global)
    zm = _standardize_for_corr(pred_resid_meso)
    zx = _standardize_for_corr(pred_resid_xylem)
    return torch.mean(zg * zb) ** 2 + torch.mean(zg * zm) ** 2 + torch.mean(zg * zx) ** 2


def build_prop_weights(y_prop_train, celltypes):
    mean_prop = y_prop_train.mean(axis=0)
    w = 1.0 / np.power(mean_prop + PROP_WEIGHT_EPS, PROP_WEIGHT_POWER)
    w = w / w.mean()
    w = np.clip(w, PROP_WEIGHT_CLIP_MIN, PROP_WEIGHT_CLIP_MAX).astype(np.float32)
    return w, {ct: float(x) for ct, x in zip(celltypes, w)}


def safe_pearsonr(x, y):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def masked_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))


def lin_ccc(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    if len(y_true) < 3:
        return np.nan
    mt = np.mean(y_true); mp = np.mean(y_pred)
    vt = np.var(y_true); vp = np.var(y_pred)
    cov = np.mean((y_true - mt) * (y_pred - mp))
    return float(2 * cov / (vt + vp + (mt - mp) ** 2 + 1e-12))


def train_one_epoch(model, loader, optimizer, loss_prop_fn, device, loss_weights):
    model.train()
    totals = {k: 0.0 for k in ["loss", "prop_loss", "burden_loss", "global_loss", "resid_meso_loss", "resid_xylem_loss", "decor_loss"]}
    n = 0
    mse = nn.MSELoss()

    for Xb, tb, ypb, ybb, ygb, yrmb, yrxb in loader:
        Xb = Xb.to(device); tb = tb.to(device); ypb = ypb.to(device)
        ybb = ybb.to(device); ygb = ygb.to(device); yrmb = yrmb.to(device); yrxb = yrxb.to(device)
        optimizer.zero_grad()
        pp, pb, pg, pm, px = model(Xb, tb)
        lp = loss_prop_fn(pp, ypb, tb)
        lb = mse(pb, ybb)
        lg = mse(pg, ygb)
        lm = masked_mse_loss(pm, yrmb)
        lx = masked_mse_loss(px, yrxb)
        ld = selective_decorrelation_loss(pb, pg, pm, px)
        loss = (
            loss_weights["prop"] * lp + loss_weights["burden"] * lb + loss_weights["global"] * lg +
            loss_weights["resid_meso"] * lm + loss_weights["resid_xylem"] * lx + loss_weights["decor"] * ld
        )
        loss.backward(); optimizer.step()
        bs = Xb.size(0); n += bs
        vals = {"loss": loss, "prop_loss": lp, "burden_loss": lb, "global_loss": lg,
                "resid_meso_loss": lm, "resid_xylem_loss": lx, "decor_loss": ld}
        for k, v in vals.items():
            totals[k] += float(v.item()) * bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def eval_one_epoch(model, loader, loss_prop_fn, device, loss_weights):
    model.eval(); mse = nn.MSELoss(); n = 0
    totals = {k: 0.0 for k in ["loss", "prop_loss", "burden_loss", "global_loss", "resid_meso_loss", "resid_xylem_loss", "decor_loss"]}
    store = {k: [] for k in ["pred_prop", "true_prop", "pred_burden", "true_burden", "pred_global", "true_global",
                              "pred_resid_meso", "true_resid_meso", "pred_resid_xylem", "true_resid_xylem", "time_idx"]}

    for Xb, tb, ypb, ybb, ygb, yrmb, yrxb in loader:
        Xb = Xb.to(device); tb = tb.to(device); ypb = ypb.to(device)
        ybb = ybb.to(device); ygb = ygb.to(device); yrmb = yrmb.to(device); yrxb = yrxb.to(device)
        pp, pb, pg, pm, px = model(Xb, tb)
        lp = loss_prop_fn(pp, ypb, tb); lb = mse(pb, ybb); lg = mse(pg, ygb)
        lm = masked_mse_loss(pm, yrmb); lx = masked_mse_loss(px, yrxb)
        ld = selective_decorrelation_loss(pb, pg, pm, px)
        loss = (
            loss_weights["prop"] * lp + loss_weights["burden"] * lb + loss_weights["global"] * lg +
            loss_weights["resid_meso"] * lm + loss_weights["resid_xylem"] * lx + loss_weights["decor"] * ld
        )
        bs = Xb.size(0); n += bs
        vals = {"loss": loss, "prop_loss": lp, "burden_loss": lb, "global_loss": lg,
                "resid_meso_loss": lm, "resid_xylem_loss": lx, "decor_loss": ld}
        for k, v in vals.items(): totals[k] += float(v.item()) * bs

        store["pred_prop"].append(pp.cpu().numpy()); store["true_prop"].append(ypb.cpu().numpy())
        store["pred_burden"].append(pb.cpu().numpy()); store["true_burden"].append(ybb.cpu().numpy())
        store["pred_global"].append(pg.cpu().numpy()); store["true_global"].append(ygb.cpu().numpy())
        store["pred_resid_meso"].append(pm.cpu().numpy()); store["true_resid_meso"].append(yrmb.cpu().numpy())
        store["pred_resid_xylem"].append(px.cpu().numpy()); store["true_resid_xylem"].append(yrxb.cpu().numpy())
        store["time_idx"].append(tb.cpu().numpy())

    out = {k: v / n for k, v in totals.items()}
    for k in ["pred_prop", "true_prop", "pred_burden", "true_burden", "pred_global", "true_global",
              "pred_resid_meso", "true_resid_meso", "pred_resid_xylem", "true_resid_xylem"]:
        out[k] = np.vstack(store[k])
    out["pred_burden"] = out["pred_burden"].ravel(); out["true_burden"] = out["true_burden"].ravel()
    out["pred_global"] = out["pred_global"].ravel(); out["true_global"] = out["true_global"].ravel()
    out["pred_resid_meso"] = out["pred_resid_meso"].ravel(); out["true_resid_meso"] = out["true_resid_meso"].ravel()
    out["pred_resid_xylem"] = out["pred_resid_xylem"].ravel(); out["true_resid_xylem"] = out["true_resid_xylem"].ravel()
    out["time_idx"] = np.concatenate(store["time_idx"]).ravel()
    return out


def compute_metrics(stats, celltypes):
    pred_prop = stats["pred_prop"]; true_prop = stats["true_prop"]; time_idx = stats["time_idx"]
    idx_to_time = {v: k for k, v in TIME_TO_INDEX.items()}
    metrics = {
        "burden_rmse": masked_rmse(stats["true_burden"], stats["pred_burden"]),
        "burden_pearson": safe_pearsonr(stats["true_burden"], stats["pred_burden"]),
        "burden_ccc": lin_ccc(stats["true_burden"], stats["pred_burden"]),
        "global_rmse": masked_rmse(stats["true_global"], stats["pred_global"]),
        "global_pearson": safe_pearsonr(stats["true_global"], stats["pred_global"]),
        "global_ccc": lin_ccc(stats["true_global"], stats["pred_global"]),
        "resid_meso_rmse": masked_rmse(stats["true_resid_meso"], stats["pred_resid_meso"]),
        "resid_meso_pearson": safe_pearsonr(stats["true_resid_meso"], stats["pred_resid_meso"]),
        "resid_meso_ccc": lin_ccc(stats["true_resid_meso"], stats["pred_resid_meso"]),
        "resid_xylem_rmse": masked_rmse(stats["true_resid_xylem"], stats["pred_resid_xylem"]),
        "resid_xylem_pearson": safe_pearsonr(stats["true_resid_xylem"], stats["pred_resid_xylem"]),
        "resid_xylem_ccc": lin_ccc(stats["true_resid_xylem"], stats["pred_resid_xylem"]),
        "prop_mse": float(np.mean((pred_prop - true_prop) ** 2)),
    }

    ct_rows = []
    for i, ct in enumerate(celltypes):
        yt = true_prop[:, i]; yp = pred_prop[:, i]
        ct_rows.append({"celltype": ct, "pearson": safe_pearsonr(yt, yp), "rmse": masked_rmse(yt, yp),
                        "ccc": lin_ccc(yt, yp), "n_samples": int(len(yt))})
    metrics["celltype_metrics"] = ct_rows
    metrics["celltype_mean_pearson"] = float(np.nanmean([r["pearson"] for r in ct_rows]))
    metrics["celltype_mean_rmse"] = float(np.nanmean([r["rmse"] for r in ct_rows]))
    metrics["celltype_mean_ccc"] = float(np.nanmean([r["ccc"] for r in ct_rows]))

    tc_rows = []
    for t in sorted(np.unique(time_idx)):
        mask = time_idx == t
        for i, ct in enumerate(celltypes):
            yt = true_prop[mask, i]; yp = pred_prop[mask, i]
            tc_rows.append({"time_idx": int(t), "time_str": idx_to_time[int(t)], "celltype": ct,
                            "pearson": safe_pearsonr(yt, yp), "rmse": masked_rmse(yt, yp),
                            "ccc": lin_ccc(yt, yp), "n_samples": int(len(yt))})
    metrics["time_celltype_metrics"] = tc_rows

    time_rows = []
    for t in sorted(np.unique(time_idx)):
        rows = [r for r in tc_rows if r["time_idx"] == int(t)]
        time_rows.append({
            "time_idx": int(t), "time_str": idx_to_time[int(t)],
            "celltype_mean_ccc": float(np.nanmean([r["ccc"] for r in rows])),
            "celltype_mean_pearson": float(np.nanmean([r["pearson"] for r in rows])),
            "celltype_mean_rmse": float(np.nanmean([r["rmse"] for r in rows])),
            "n_celltypes": len(rows),
        })
    metrics["time_metrics"] = time_rows
    return metrics


# =============================================================================
# 7. Load generated split pseudo-spots for model training
# =============================================================================
def load_generated_split(split_name: str, celltypes: List[str]):
    Xs = []; ts = []; props = []; burdens = []; globals_ = []; rms = []; rxs = []; metas = []
    prop_cols = [f"prop__{ct}" for ct in celltypes]

    for t in TIME_POINTS:
        d = os.path.join(PSEUDO_DIR, split_name)
        X = np.load(os.path.join(d, f"X_pseudospots_{t}.npy")).astype(np.float32)
        meta = pd.read_csv(os.path.join(d, f"meta_pseudospots_{t}.csv"))
        if len(meta) != X.shape[0]:
            raise ValueError(f"Row mismatch {split_name}/{t}")
        Xs.append(X)
        ts.append(np.full(len(meta), TIME_TO_INDEX[t], dtype=np.int64))
        props.append(meta[prop_cols].to_numpy(dtype=np.float32))
        burdens.append(meta[[BURDEN_COL]].to_numpy(dtype=np.float32))
        globals_.append(meta[[GLOBAL_COL]].to_numpy(dtype=np.float32))
        rms.append(meta[[RESID_MESO_COL]].to_numpy(dtype=np.float32))
        rxs.append(meta[[RESID_XYLEM_COL]].to_numpy(dtype=np.float32))
        metas.append(meta)

    return (
        np.vstack(Xs), np.concatenate(ts), np.vstack(props), np.vstack(burdens), np.vstack(globals_),
        np.vstack(rms), np.vstack(rxs), pd.concat(metas, ignore_index=True)
    )


# =============================================================================
# 8. Main training procedure
# =============================================================================
def train_model_on_corrected_splits():
    celltypes = pd.read_csv(os.path.join(PSEUDO_DIR, "celltypes_v2_residual.csv"), header=None).iloc[:, 0].astype(str).tolist()

    train_data = load_generated_split("train", celltypes)
    val_data = load_generated_split("validation", celltypes)
    test_data = load_generated_split("test", celltypes)

    X_train, time_train, y_prop_train, y_burden_train, y_global_train, y_rm_train, y_rx_train, meta_train = train_data
    X_val, time_val, y_prop_val, y_burden_val, y_global_val, y_rm_val, y_rx_val, meta_val = val_data
    X_test, time_test, y_prop_test, y_burden_test, y_global_test, y_rm_test, y_rx_test, meta_test = test_data

    log("\n[Generated pseudo-spot split sizes]")
    log(f"Train      : {len(X_train)}")
    log(f"Validation : {len(X_val)}")
    log(f"Test       : {len(X_test)}")
    log("Test time counts:\n" + str(meta_test["time_str"].value_counts().sort_index()))

    # Train-only preprocessing.
    scaler = StandardScaler(with_mean=STANDARD_SCALER_WITH_MEAN)
    X_train_scaled = np.asarray(scaler.fit_transform(X_train), dtype=np.float32)
    X_val_scaled = np.asarray(scaler.transform(X_val), dtype=np.float32)
    X_test_scaled = np.asarray(scaler.transform(X_test), dtype=np.float32)
    with open(os.path.join(OUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    prop_weights_np, prop_weight_dict = build_prop_weights(y_prop_train, celltypes)
    prop_weights_t = torch.tensor(prop_weights_np, dtype=torch.float32, device=DEVICE)
    time_weights_np = np.asarray([TIME_PROP_WEIGHTS[t] for t in TIME_POINTS], dtype=np.float32)
    time_weights_t = torch.tensor(time_weights_np, dtype=torch.float32, device=DEVICE)
    loss_prop_fn = WeightedPropMSELoss(prop_weights_t, time_weights_t)

    train_ds = TIDESTDataset(X_train_scaled, time_train, y_prop_train, y_burden_train, y_global_train, y_rm_train, y_rx_train)
    val_ds = TIDESTDataset(X_val_scaled, time_val, y_prop_val, y_burden_val, y_global_val, y_rm_val, y_rx_val)
    test_ds = TIDESTDataset(X_test_scaled, time_test, y_prop_test, y_burden_test, y_global_test, y_rm_test, y_rx_test)

    fixed_model_params = {
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "dropout": 0.2, "time_hidden_dim": 16,
    }
    default_lambda_params = {
        "lambda_burden": LAMBDA_BURDEN, "lambda_global": LAMBDA_GLOBAL,
        "lambda_resid": LAMBDA_RESID, "lambda_decor": LAMBDA_DECOR,
    }

    def make_loss_weights(p):
        return {
            "prop": LAMBDA_PROP,
            "burden": float(p["lambda_burden"]),
            "global": float(p["lambda_global"]),
            "resid_meso": float(p["lambda_resid"]),
            "resid_xylem": float(p["lambda_resid"]),
            "decor": float(p["lambda_decor"]),
        }

    def objective(trial):
        # Same seed across trials isolates the effect of hyperparameters.
        set_seed(TRAINING_SEED)
        p = {
            "lambda_burden": trial.suggest_categorical("lambda_burden", LAMBDA_SEARCH_VALUES),
            "lambda_global": trial.suggest_categorical("lambda_global", LAMBDA_SEARCH_VALUES),
            "lambda_resid": trial.suggest_categorical("lambda_resid", LAMBDA_SEARCH_VALUES),
            "lambda_decor": trial.suggest_categorical("lambda_decor", LAMBDA_SEARCH_VALUES),
        }
        weights = make_loss_weights(p)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        model = TIDESTModel(X_train.shape[1], len(celltypes), N_TIME, 16, 0.2).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        best_ccc = -np.inf; best_loss = np.inf; best_epoch = -1; patience = 0
        try:
            for epoch in range(1, SEARCH_NUM_EPOCHS + 1):
                train_one_epoch(model, train_loader, optimizer, loss_prop_fn, DEVICE, weights)
                vs = eval_one_epoch(model, val_loader, loss_prop_fn, DEVICE, weights)
                vm = compute_metrics(vs, celltypes)
                score = float(vm["celltype_mean_ccc"])
                if not np.isfinite(score): score = -1.0
                improved = score > best_ccc + 1e-8 or (abs(score - best_ccc) <= 1e-8 and vs["loss"] < best_loss)
                if improved:
                    best_ccc = score; best_loss = float(vs["loss"]); best_epoch = epoch; patience = 0
                else:
                    patience += 1
                trial.report(best_ccc, step=epoch)
                if trial.should_prune(): raise optuna.TrialPruned()
                if patience >= SEARCH_PATIENCE: break
            trial.set_user_attr("best_val_loss", best_loss)
            trial.set_user_attr("best_epoch", best_epoch)
            return best_ccc
        finally:
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    selected = default_lambda_params.copy()
    if RUN_HPARAM_SEARCH:
        db_path = os.path.join(OUT_DIR, "optuna_lambda_discrete_source_split.db")
        if RESET_OPTUNA_STUDY and os.path.exists(db_path):
            os.remove(db_path)
            log(f"[RESET] removed old Optuna DB: {db_path}")
        storage = f"sqlite:///{Path(db_path).as_posix()}"
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=TRAINING_SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10, interval_steps=1),
            load_if_exists=True,
        )
        n_remaining = max(0, N_TRIALS - len(study.trials))
        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining, gc_after_trial=True)
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            raise RuntimeError("Optuna search produced no completed trial.")
        selected.update(study.best_params)
        study.trials_dataframe().to_csv(os.path.join(OUT_DIR, "optuna_trials.csv"), index=False)
        with open(os.path.join(OUT_DIR, "optuna_best_params.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_value_val_celltype_mean_ccc": float(study.best_value),
                "best_trial_number": int(study.best_trial.number),
                "best_trial_best_epoch": int(study.best_trial.user_attrs.get("best_epoch", -1)),
                "best_lambda_params": selected,
                "search_space": LAMBDA_SEARCH_VALUES,
            }, f, indent=2, ensure_ascii=False)
        log(f"[Optuna best] val CCC={study.best_value:.6f} | params={selected}")

    final_weights = make_loss_weights(selected)
    set_seed(TRAINING_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = TIDESTModel(X_train.shape[1], len(celltypes), N_TIME, 16, 0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_path = os.path.join(OUT_DIR, "best_model.pt")
    best_ccc = -np.inf; best_loss = np.inf; best_epoch = -1; patience = 0; history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimizer, loss_prop_fn, DEVICE, final_weights)
        vs = eval_one_epoch(model, val_loader, loss_prop_fn, DEVICE, final_weights)
        vm = compute_metrics(vs, celltypes)
        row = {
            "epoch": epoch, "train_loss": tr["loss"], "val_loss": vs["loss"],
            "val_celltype_mean_ccc": vm["celltype_mean_ccc"],
            "val_celltype_mean_pearson": vm["celltype_mean_pearson"],
            "val_celltype_mean_rmse": vm["celltype_mean_rmse"],
        }
        history.append(row)
        log(f"[Epoch {epoch:03d}] train={tr['loss']:.6f} val={vs['loss']:.6f} val_macro_CCC={vm['celltype_mean_ccc']:.4f}")

        improved = row["val_celltype_mean_ccc"] > best_ccc + 1e-8 or (
            abs(row["val_celltype_mean_ccc"] - best_ccc) <= 1e-8 and row["val_loss"] < best_loss
        )
        if improved:
            best_ccc = float(row["val_celltype_mean_ccc"]); best_loss = float(row["val_loss"])
            best_epoch = epoch; patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            log(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
    if best_epoch < 0:
        raise RuntimeError("No valid best checkpoint saved.")

    # -------------------------------------------------------------------------
    # ONE-TIME held-out TEST evaluation: test did not affect search/checkpoint.
    # -------------------------------------------------------------------------
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    ts = eval_one_epoch(model, test_loader, loss_prop_fn, DEVICE, final_weights)
    tm = compute_metrics(ts, celltypes)

    pred_df = meta_test.copy()
    pred_df["time_idx"] = ts["time_idx"]
    pred_df["pred_infection_burden"] = ts["pred_burden"]
    pred_df["true_infection_burden"] = ts["true_burden"]
    pred_df["pred_global_program_score"] = ts["pred_global"]
    pred_df["true_global_program_score"] = ts["true_global"]
    pred_df["pred_residual_meso"] = ts["pred_resid_meso"]
    pred_df["true_residual_meso"] = ts["true_resid_meso"]
    pred_df["pred_residual_xylem"] = ts["pred_resid_xylem"]
    pred_df["true_residual_xylem"] = ts["true_resid_xylem"]
    for i, ct in enumerate(celltypes):
        pred_df[f"pred_prop__{ct}"] = ts["pred_prop"][:, i]
        pred_df[f"true_prop__{ct}"] = ts["true_prop"][:, i]
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    with open(os.path.join(OUT_DIR, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(tm, f, indent=2, ensure_ascii=False)
    pd.DataFrame(tm["celltype_metrics"]).to_csv(os.path.join(OUT_DIR, "test_celltype_metrics.csv"), index=False)
    pd.DataFrame(tm["time_celltype_metrics"]).to_csv(os.path.join(OUT_DIR, "test_time_celltype_metrics.csv"), index=False)
    pd.DataFrame(tm["time_metrics"]).to_csv(os.path.join(OUT_DIR, "test_time_metrics.csv"), index=False)

    config = {
        "version": "TIDE-ST_v5.8_source_split721_end_to_end",
        "source_split_before_pseudospot_generation": True,
        "source_split_seed": DATA_PARTITION_SEED,
        "source_split_fractions": {"train": TRAIN_FRACTION, "validation": VAL_FRACTION, "test": TEST_FRACTION},
        "pseudospot_total_per_time": TOTAL_PSEUDOSPOTS_PER_TIME,
        "pseudospot_seed": PSEUDOSPOT_SEED,
        "training_seed": TRAINING_SEED,
        "preprocessing_fit_on": "train_pseudospots_only",
        "auxiliary_target_definition_fit_on": "train_source_profiles_only",
        "hyperparameter_search_uses_test": False,
        "checkpoint_selection_uses_test": False,
        "selected_lambda_params": selected,
        "final_loss_weights": final_weights,
        "prop_weights": prop_weight_dict,
        "time_prop_weights": TIME_PROP_WEIGHTS,
        "standard_scaler_with_mean": STANDARD_SCALER_WITH_MEAN,
        "best_epoch": best_epoch,
        "best_val_celltype_mean_ccc": best_ccc,
        "held_out_test_summary": {
            "celltype_mean_ccc": tm["celltype_mean_ccc"],
            "celltype_mean_pearson": tm["celltype_mean_pearson"],
            "celltype_mean_rmse": tm["celltype_mean_rmse"],
        },
    }
    with open(os.path.join(OUT_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 80)
    log("TRUE HELD-OUT TEST RESULTS")
    log("=" * 80)
    log(f"celltype_mean_ccc     = {tm['celltype_mean_ccc']:.4f}")
    log(f"celltype_mean_pearson = {tm['celltype_mean_pearson']:.4f}")
    log(f"celltype_mean_rmse    = {tm['celltype_mean_rmse']:.4f}")
    log(f"burden_ccc            = {tm['burden_ccc']:.4f}")
    log(f"global_ccc            = {tm['global_ccc']:.4f}")
    log(f"resid_meso_ccc        = {tm['resid_meso_ccc']:.4f}")
    log(f"resid_xylem_ccc       = {tm['resid_xylem_ccc']:.4f}")
    log(f"Outputs: {OUT_DIR}")


# =============================================================================
# 9. End-to-end main
# =============================================================================
def main():
    if not np.isclose(TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION, 1.0):
        raise ValueError("TRAIN_FRACTION + VAL_FRACTION + TEST_FRACTION must equal 1.0")

    set_seed(DATA_PARTITION_SEED)
    log("=" * 88)
    log("TIDE-ST v5.8 | SOURCE-PROFILE 70:20:10 SPLIT BEFORE PSEUDO-SPOT GENERATION")
    log("=" * 88)
    log(f"Device: {DEVICE}")

    # 1) Read-only source loading + source-level split.
    adatas = load_source_adatas()
    assign_source_splits(adatas)

    # 2) Refit all target definitions/scaling on TRAIN source profiles only.
    log("\n" + "=" * 88)
    log("FIT AUXILIARY TARGET DEFINITIONS ON TRAIN SOURCE PROFILES ONLY")
    log("=" * 88)
    fit_all_auxiliary_targets(adatas)

    # 3) Save leakage-free split references for baseline methods.
    save_split_h5ads(adatas)

    # 4) Generate split-specific pseudo-spots independently.
    log("\n" + "=" * 88)
    log("GENERATE TRAIN / VALIDATION / TEST PSEUDO-SPOTS INDEPENDENTLY")
    log("=" * 88)
    pseudospots = generate_all_split_pseudospots(adatas)
    build_test_scenario_table(pseudospots)

    # Release source adatas before GPU training.
    del adatas
    gc.collect()

    # 5) Train/validate/test model without any pseudo-spot-level re-splitting.
    log("\n" + "=" * 88)
    log("TRAIN TIDE-ST USING THE PRE-GENERATED SOURCE-DISJOINT SPLITS")
    log("=" * 88)
    train_model_on_corrected_splits()

    log("\nDONE.")
    log("Key audit files:")
    log(os.path.join(PSEUDO_DIR, "source_profile_leakage_audit.json"))
    log(os.path.join(PSEUDO_DIR, "pseudospot_source_overlap_audit.json"))
    log(os.path.join(PSEUDO_DIR, "source_profile_split_7_2_1.csv"))
    log(os.path.join(PSEUDO_DIR, "Table_S1_candidate_test_counts.csv"))


if __name__ == "__main__":
    main()
