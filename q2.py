# compare_gd_momentum_nag.py
"""
Compare simple gradient descent, momentum GD, and NAG on up to 3 datasets.
- Reads CSVs from current dir (or specified paths).
- Expects a label column: 'price' for housing, 'Sales' for advertising, or detects 'target','label'.
- Uses full-batch (deterministic) gradient descent variants.
- Measures epochs & wall-time needed to reach within tol=1% of closed-form optimal MSE.
"""

import os
import time
import glob
import math
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Utilities: data handling
# -------------------------
def read_csv_safe(path: str) -> pd.DataFrame:
    """Read CSV robustly. Attempts pandas default, then python engine."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine='python')

def preprocess_for_regression(df: pd.DataFrame, target_candidates=['price','Price','Sales','sales','target','label']):
    """
    - Detect target column from target_candidates (first matching), remove it from features.
    - Map yes/no -> 1/0 and one-hot encode other categoricals.
    - Fill NaNs with column mean.
    - Standardize features (zero mean, unit std).
    Returns: X (m x p), y (m,), feature_names, scaler dict (mean,std)
    """
    df = df.copy()
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # detect target
    target_col = None
    for cand in target_candidates:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        # fall back: last column as target
        target_col = df.columns[-1]
        print(f"Warning: no standard target column found. Using last column '{target_col}' as target.")

    y = pd.to_numeric(df[target_col], errors='coerce')
    Xdf = df.drop(columns=[target_col])

    # map yes/no to 1/0
    for c in Xdf.columns:
        if Xdf[c].dtype == object:
            low = Xdf[c].astype(str).str.strip().str.lower()
            if set(low.unique()) <= {'yes','no'} or ('yes' in set(low.unique()) or 'no' in set(low.unique())):
                Xdf[c] = low.map({'yes':1,'no':0})

    # one-hot encode remaining object columns
    obj_cols = Xdf.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        Xdf = pd.get_dummies(Xdf, columns=obj_cols, dummy_na=False)

    Xdf = Xdf.apply(pd.to_numeric, errors='coerce')
    # fill NaNs with column mean
    Xdf = Xdf.fillna(Xdf.mean())

    # Drop any columns with zero variance (avoid singular X^T X)
    variances = Xdf.var(axis=0)
    zero_var_cols = variances[variances <= 0].index.tolist()
    if zero_var_cols:
        Xdf = Xdf.drop(columns=zero_var_cols)
        print("Dropped zero-variance columns:", zero_var_cols)

    X = Xdf.values.astype(float)
    y = y.fillna(y.mean()).values.astype(float)

    # standardize X (feature-wise), store means and stds
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std_adj = std.copy()
    std_adj[std_adj == 0] = 1.0
    Xs = (X - mean) / std_adj

    return Xs, y, Xdf.columns.tolist(), {'mean': mean, 'std': std_adj}, target_col

# -------------------------
# Closed-form optimal
# -------------------------
def closed_form_solution(X: np.ndarray, y: np.ndarray, ridge=1e-8):
    m, p = X.shape
    # include bias via augmentation
    Xb = np.hstack([np.ones((m,1)), X])
    # closed-form with ridge on weights (not bias)
    I = np.eye(p+1)
    I[0,0] = 0.0
    w_opt = np.linalg.solve(Xb.T @ Xb + ridge * I, Xb.T @ y)
    # compute MSE
    preds = Xb @ w_opt
    mse = np.mean((y - preds)**2)
    return w_opt, mse

# -------------------------
# Loss and gradient (MSE)
# -------------------------
def mse_and_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    w includes bias as first element.
    X is m x p (standardized features).
    """
    m, p = X.shape
    Xb = np.hstack([np.ones((m,1)), X])  # m x (p+1)
    preds = Xb @ w
    err = preds - y
    loss = float(np.mean(err**2))
    grad = (2.0 / m) * (Xb.T @ err)  # shape (p+1,)
    return loss, grad

# -------------------------
# Optimizers
# -------------------------
def train_gd(X, y, w0, lr=0.01, max_epochs=2000, tol_epochs=0.0, w_opt=None, verbose=False):
    w = w0.copy()
    losses = []
    t0 = time.time()
    for epoch in range(1, max_epochs+1):
        loss, grad = mse_and_grad(w, X, y)
        w -= lr * grad
        losses.append(loss)
        if verbose and (epoch % 500 == 0):
            print(f"[GD] epoch {epoch}, loss={loss:.6f}")
        # stopping check will be handled outside using w_opt and tol
    duration = time.time() - t0
    return w, losses, duration

def train_momentum(X, y, w0, lr=0.01, gamma=0.9, max_epochs=2000, verbose=False):
    w = w0.copy()
    v = np.zeros_like(w)
    losses = []
    t0 = time.time()
    for epoch in range(1, max_epochs+1):
        loss, grad = mse_and_grad(w, X, y)
        # vt+1 = gamma vt + eta * grad(wt)
        v = gamma * v + lr * grad
        # wt+1 = wt - vt+1
        w = w - v
        losses.append(loss)
        if verbose and (epoch % 500 == 0):
            print(f"[Momentum] epoch {epoch}, loss={loss:.6f}")
    duration = time.time() - t0
    return w, losses, duration

def train_nag(X, y, w0, lr=0.01, gamma=0.9, max_epochs=2000, verbose=False):
    w = w0.copy()
    v = np.zeros_like(w)
    losses = []
    t0 = time.time()
    for epoch in range(1, max_epochs+1):
        # compute gradient at lookahead: wt - gamma vt
        w_look = w - gamma * v
        # grad at lookahead
        loss_look, grad_look = mse_and_grad(w_look, X, y)
        # vt+1 = gamma vt + eta * grad(wt - gamma vt)
        v = gamma * v + lr * grad_look
        # wt+1 = wt - vt+1
        w = w - v
        # compute true loss at new w (for logging)
        loss, _ = mse_and_grad(w, X, y)
        losses.append(loss)
        if verbose and (epoch % 500 == 0):
            print(f"[NAG] epoch {epoch}, loss={loss:.6f}")
    duration = time.time() - t0
    return w, losses, duration

# -------------------------
# Runner to compare methods on one dataset
# -------------------------
def compare_on_dataset(X, y, dataset_name="dataset", n_epochs=2000, lr=0.01, gamma=0.9, tol_rel=0.01):
    m, p = X.shape
    print(f"\n--- Dataset: {dataset_name} (m={m}, p={p}) ---")
    # closed-form optimal
    w_opt, mse_opt = closed_form_solution(X, y, ridge=1e-8)
    print(f"Closed-form optimal MSE: {mse_opt:.8f}")

    # initialization
    rng = np.random.RandomState(0)
    w0 = rng.randn(p+1) * 0.01

    results = {}

    # Simple GD
    w_gd, losses_gd, dur_gd = train_gd(X, y, w0, lr=lr, max_epochs=n_epochs, verbose=False)
    # Momentum
    w_m, losses_m, dur_m = train_momentum(X, y, w0, lr=lr, gamma=gamma, max_epochs=n_epochs, verbose=False)
    # NAG
    w_nag, losses_nag, dur_nag = train_nag(X, y, w0, lr=lr, gamma=gamma, max_epochs=n_epochs, verbose=False)

    # compute epoch when within tol_rel of optimal (relative)
    def epochs_to_tol(losses, L_opt, tol_rel):
        target = L_opt * (1.0 + tol_rel)
        for i, L in enumerate(losses, 1):
            if L <= target:
                return i
        return None

    e_gd = epochs_to_tol(losses_gd, mse_opt, tol_rel)
    e_m = epochs_to_tol(losses_m, mse_opt, tol_rel)
    e_nag = epochs_to_tol(losses_nag, mse_opt, tol_rel)

    results['gd'] = {'losses': losses_gd, 'time': dur_gd, 'epochs_to_tol': e_gd, 'final_loss': losses_gd[-1]}
    results['momentum'] = {'losses': losses_m, 'time': dur_m, 'epochs_to_tol': e_m, 'final_loss': losses_m[-1]}
    results['nag'] = {'losses': losses_nag, 'time': dur_nag, 'epochs_to_tol': e_nag, 'final_loss': losses_nag[-1]}

    # print summary
    print("Method    | epochs_to_1%opt | final_loss        | time(s)")
    for name in ['gd','momentum','nag']:
        row = results[name]
        print(f"{name:9s} | {str(row['epochs_to_tol']).rjust(14)} | {row['final_loss']:17.8f} | {row['time']:.4f}")

    # plot losses
    plt.figure(figsize=(7,4))
    max_len = max(len(results['gd']['losses']), len(results['momentum']['losses']), len(results['nag']['losses']))
    plt.plot(results['gd']['losses'], label='GD')
    plt.plot(results['momentum']['losses'], label='Momentum')
    plt.plot(results['nag']['losses'], label='NAG')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (log scale)')
    plt.title(f'Convergence on {dataset_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results, mse_opt

# -------------------------
# Main: run on up to three datasets
# -------------------------
if __name__ == "__main__":
    # find CSVs in current directory
    csvs = sorted(glob.glob("*.csv"))
    print("CSV files found:", csvs)
    chosen = []
    # Prefer advertising.csv and housing.csv if present
    for preferred in ['advertising.csv', 'housing.csv']:
        if preferred in csvs:
            chosen.append(preferred)
    # Add other CSVs until we have 3
    for c in csvs:
        if c not in chosen and len(chosen) < 3:
            chosen.append(c)
    # If fewer than 3, create synthetic datasets
    synth_added = 0
    while len(chosen) < 3:
        chosen.append(None)  # placeholder for synthetic
        synth_added += 1

    print("Datasets to process (None => synthetic):", chosen)

    overall_summary = []
    for idx, path in enumerate(chosen):
        if path is None:
            # create a synthetic dataset (low-rank + noise) for testing
            m = 200
            p = 20
            rng = np.random.RandomState(100 + idx)
            U = rng.randn(m, 5)
            V = rng.randn(5, p)
            X = U.dot(V) + 0.1 * rng.randn(m, p)
            # create a target linearly
            true_w = rng.randn(p)
            y = X.dot(true_w) + 0.5 * rng.randn(m)
            name = f"synthetic_{idx}"
            # standardize
            Xs = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-10)
        else:
            df = read_csv_safe(path)
            Xs, y, feature_names, scaler, target_col = preprocess_for_regression(df)
            name = path

        # compare (you can tune lr/gamma as required)
        results, mse_opt = compare_on_dataset(Xs, y, dataset_name=name, n_epochs=2000, lr=0.01, gamma=0.9, tol_rel=0.01)
        overall_summary.append((name, mse_opt, results))

    # Save a CSV summary
    rows = []
    for name, mse_opt, res in overall_summary:
        for method in ['gd','momentum','nag']:
            rows.append({
                'dataset': name,
                'method': method,
                'mse_opt': mse_opt,
                'final_loss': res[method]['final_loss'],
                'time_s': res[method]['time'],
                'epochs_to_1pct_opt': res[method]['epochs_to_tol']
            })
    pd.DataFrame(rows).to_csv('gd_momentum_nag_summary.csv', index=False)
    print("\nSaved overall summary to gd_momentum_nag_summary.csv")
