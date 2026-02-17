from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import joblib


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def group_split_idx(
    df: pd.DataFrame,
    label_col: str,
    seed: int = 42,
    test_size: float = 0.2,
):
    """Safe group split even if only one session exists."""
    if df["session_id"].nunique() < 2:
        idx = np.arange(len(df))
        split = int(len(idx) * (1 - test_size))
        return idx[:split], idx[split:]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X = df.drop(columns=[label_col])
    y = df[label_col]
    tr, te = next(gss.split(X, y, groups=df["session_id"]))
    return tr, te


def build_preprocessor(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


def build_model(name: str, seed: int):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    raise ValueError(name)


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# -----------------------------
# Evaluation
# -----------------------------

def eval_task(
    task: str,
    df: pd.DataFrame,
    label_col: str,
    features: List[str],
    outdir: str,
    seed: int,
) -> List[Dict]:

    ensure_dir(outdir)
    df = df[["session_id", label_col] + features].dropna(subset=[label_col])

    tr_idx, te_idx = group_split_idx(df, label_col, seed)
    tr, te = df.iloc[tr_idx], df.iloc[te_idx]

    X_tr = tr.drop(columns=[label_col, "session_id"])
    y_tr = tr[label_col].astype(str)
    X_te = te.drop(columns=[label_col, "session_id"])
    y_te = te[label_col].astype(str)

    rows = []

    for model_name in ["majority", "logreg", "rf"]:
        pipe = Pipeline([
            ("pre", build_preprocessor(X_tr)),
            ("clf", build_model(model_name, seed)),
        ])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        rows.append({
            "task": task,
            "model": model_name,
            "macro_f1": macro_f1(y_te, pred),
            "acc": float(accuracy_score(y_te, pred)),
            "n_train": len(X_tr),
            "n_test": len(X_te),
        })

        if model_name != "majority":
            joblib.dump(pipe, os.path.join(outdir, f"{task}_{model_name}.joblib"))

    return rows


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", default="fourthorse_trial_dummy_dataset.zip")
    parser.add_argument("--outdir", default="poster_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    eval_dir = os.path.join(args.outdir, "_eval_artifacts")
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dir(eval_dir)
    ensure_dir(fig_dir)

    if os.path.exists(args.zip_path):
        with zipfile.ZipFile(args.zip_path) as z:
            z.extractall("dataset_tmp")
        steps = pd.read_csv("dataset_tmp/steps.csv")
        dataset_hash = sha256_file(args.zip_path)
    else:
        steps = pd.read_csv("data/steps.csv")
        dataset_hash = sha256_str(steps.to_csv(index=False))

    features = [c for c in steps.columns if c not in ("latent_state", "session_id")]

    rows = eval_task(
        "latent_state",
        steps,
        "latent_state",
        features,
        eval_dir,
        args.seed,
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "metrics_summary.csv"), index=False)

    plt.figure()
    plt.bar(df["model"], df["macro_f1"])
    plt.ylabel("Macro-F1")
    plt.title("Latent State Model Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_model_comparison_latent_state.png"))
    plt.close()

    commitments = {
        "dataset_hash": dataset_hash,
        "feature_schema_hash": sha256_str("|".join(sorted(features))),
    }
    with open(os.path.join(args.outdir, "commitments.json"), "w") as f:
        json.dump(commitments, f, indent=2)

    with open(os.path.join(args.outdir, "privacy_spec.txt"), "w") as f:
        f.write(
            "- No raw text stored\n"
            "- No keystroke streams stored\n"
            "- Aggregated features only\n"
            "- session_id only for splitting\n"
        )

    print("DONE")


if __name__ == "__main__":
    main()

