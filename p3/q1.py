# fixed_reconstruction.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# ---------------------------
# Helper: clean numeric tokens
# ---------------------------
def clean_numeric_token(tok):
    """Return float cleaned from a token like '17.5S' -> 17.5, else NaN if not parseable."""
    if pd.isna(tok):
        return np.nan
    s = str(tok).strip()
    # Keep digits, minus and dot; remove other characters
    cleaned = re.sub(r'[^0-9\.\-]+', '', s)
    if cleaned == "" or cleaned == "." or cleaned == "-":
        return np.nan
    try:
        return float(cleaned)
    except:
        return np.nan

# ---------------------------
# Robust CSV loader + cleaner
# ---------------------------
def read_and_clean_csv(path: str) -> pd.DataFrame:
    """
    Read CSV and apply token cleaning to cells that look like numbers (handles '17.5S').
    Returns a DataFrame where numeric-like strings are converted to numbers where possible.
    """
    # Read as strings first to preserve weird tokens
    df_raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Apply cleaning: if token contains any digit, attempt to clean numeric token; else keep original string
    df_clean = df_raw.copy()
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(lambda x: clean_numeric_token(x) if re.search(r'\d', str(x)) else x)
    # Return
    return df_clean

# ---------------------------
# Preprocess dataframe -> numeric matrix X
# ---------------------------
def preprocess_dataframe(df: pd.DataFrame, drop_targets: List[str] = None) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if drop_targets is None:
        drop_targets = ['price','Price','sales','Sales','target']
    # Drop target columns (case-insensitive)
    to_drop = [c for c in df.columns if c.lower() in {t.lower() for t in drop_targets}]
    if to_drop:
        print(f" - Dropping columns: {to_drop}")
        df = df.drop(columns=to_drop)
    # Map yes/no -> 1/0 where appropriate
    for col in df.columns:
        if df[col].dtype == object:
            ser = df[col].astype(str).str.strip()
            low = ser.str.lower()
            uniques = set(low.unique())
            # If column contains yes/no values (or mostly yes/no) map them
            if 'yes' in uniques or 'no' in uniques:
                df[col] = low.map({'yes':1, 'no':0})
    # One-hot encode remaining object columns
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print(f" - One-hot encoding columns: {obj_cols}")
        df = pd.get_dummies(df, columns=obj_cols, dummy_na=False)
    # Coerce all to numeric and fill NaN with column mean
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.fillna(df_numeric.mean())
    X = df_numeric.values.astype(float)
    feature_names = df_numeric.columns.tolist()
    return X, feature_names, df_numeric

# ---------------------------
# EigenFace-style reconstruction
# ---------------------------
def compute_reconstruction_stats(X: np.ndarray, n_list: List[int]):
    m, p = X.shape
    mu = np.mean(X, axis=0)
    A = X - mu
    ATA = A.T @ A
    eigvals, eigvecs = np.linalg.eigh(ATA)  # ascending
    idx_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]
    total = eigvals.sum() if eigvals.sum() > 0 else 1.0
    stats = {}
    for n in n_list:
        n_use = min(n, p)
        E = eigvecs[:, :n_use]
        W = A @ E
        L = W @ E.T
        R = L + mu
        SSE = float(np.sum((X - R)**2))
        MSE = SSE / (m * p)
        explained = float(eigvals[:n_use].sum() / total)
        stats[n] = {"n_used": n_use, "SSE": SSE, "MSE": MSE, "explained_fraction": explained}
    return stats, eigvals, eigvecs, mu

# ---------------------------
# Main flow
# ---------------------------
if __name__ == "__main__":
    # Filenames - change if needed
    housing_path = "housing.csv"
    advertising_path = "advertising.csv"

    # Read + clean both CSVs
    print("Reading and cleaning housing.csv ...")
    housing_df_raw = read_and_clean_csv(housing_path)
    print("Reading and cleaning advertising.csv ...")
    advertising_df_raw = read_and_clean_csv(advertising_path)

    # Preprocess (drop labels inside function)
    print("\nPreprocessing housing.csv")
    X_housing, feats_housing, housing_numeric_df = preprocess_dataframe(housing_df_raw, drop_targets=['price','Price'])
    print("Preprocessing advertising.csv")
    X_adv, feats_adv, adv_numeric_df = preprocess_dataframe(advertising_df_raw, drop_targets=['Sales','sales'])

    print(f"\nHousing numeric features ({len(feats_housing)}): {feats_housing}")
    print(f"Advertising numeric features ({len(feats_adv)}): {feats_adv}")

    # Choose n values
    n_values = [1,2,3,5,7,10,15]

    # Compute stats
    stats_housing, eigvals_h, eigvecs_h, mu_h = compute_reconstruction_stats(X_housing, n_values)
    stats_adv, eigvals_a, eigvecs_a, mu_a = compute_reconstruction_stats(X_adv, n_values)

    # Prepare summary
    rows = []
    for n in n_values:
        rows.append({
            "dataset": "housing",
            "n_requested": n,
            "n_used": stats_housing[n]["n_used"],
            "SSE": stats_housing[n]["SSE"],
            "MSE": stats_housing[n]["MSE"],
            "explained_fraction": stats_housing[n]["explained_fraction"]
        })
    for n in n_values:
        rows.append({
            "dataset": "advertising",
            "n_requested": n,
            "n_used": stats_adv[n]["n_used"],
            "SSE": stats_adv[n]["SSE"],
            "MSE": stats_adv[n]["MSE"],
            "explained_fraction": stats_adv[n]["explained_fraction"]
        })

    summary_df = pd.DataFrame(rows)
    print("\nReconstruction summary:")
    print(summary_df)
    summary_df.to_csv("reconstruction_summary_housing_advertising.csv", index=False)
    print("\nSaved reconstruction_summary_housing_advertising.csv")

    # Plot SSE vs n
    plt.figure(figsize=(8,5))
    ns = n_values
    sses_h = [stats_housing[n]["SSE"] for n in ns]
    sses_a = [stats_adv[n]["SSE"] for n in ns]
    plt.plot(ns, sses_h, marker='o', label='housing')
    plt.plot(ns, sses_a, marker='o', label='advertising')
    plt.xlabel("Number of eigenvectors (n) used for reconstruction")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.title("Reconstruction error (SSE) vs n")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
