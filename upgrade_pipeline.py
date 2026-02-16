from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import joblib


# -----------------------------
# IO helpers
# -----------------------------

@dataclass
class DatasetPaths:
    steps_csv: str
    choices_csv: str
    typing_csv: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unzip_dataset(zip_path: str, extract_dir: str) -> DatasetPaths:
    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    steps_csv = os.path.join(extract_dir, "steps.csv")
    choices_csv = os.path.join(extract_dir, "choices.csv")
    typing_csv = os.path.join(extract_dir, "typing.csv")

    for p in [steps_csv, choices_csv, typing_csv]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Expected file not found after unzip: {p}")

    return DatasetPaths(steps_csv=steps_csv, choices_csv=choices_csv, typing_csv=typing_csv)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def validate_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


# -----------------------------
# Split + preprocessing
# -----------------------------

def group_split_idx(
    df: pd.DataFrame,
    label_col: str,
    group_col: str = "session_id",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    y = df[label_col]
    X = df.drop(columns=[label_col])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, te_idx = next(splitter.split(X, y, groups=df[group_col]))
    return tr_idx, te_idx


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_model(model_name: str, seed: int = 42):
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model_name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(X: pd.DataFrame, model_name: str, seed: int = 42) -> Pipeline:
    pre = build_preprocessor(X)
    clf = build_model(model_name, seed=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# -----------------------------
# One task runner
# -----------------------------

def eval_task(
    task: str,
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    outdir: str,
    seed: int,
) -> List[Dict[str, object]]:
    ensure_dir(outdir)

    # keep session_id for split, drop from features for training
    needed = ["session_id", label_col] + [c for c in feature_cols if c in df.columns]
    df2 = df[needed].dropna(subset=[label_col]).copy()

    # SAFE split: group split if >=2 sessions, else row split fallback
if "session_id" in df2.columns and df2["session_id"].nunique() >= 2:
    tr_idx, te_idx = group_split_idx(df2, label_col=label_col, group_col="session_id", seed=seed)
else:
    idx = np.arange(len(df2))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    tr = df2.iloc[tr_idx].copy()
    te = df2.iloc[te_idx].copy()

    y_tr = tr[label_col].astype(str)
    y_te = te[label_col].astype(str)

    X_tr = tr.drop(columns=[label_col, "session_id"])
    X_te = te.drop(columns=[label_col, "session_id"])

    rows = []
    for model_name in ["majority", "logreg", "rf"]:
        pipe = build_pipeline(X_tr, model_name=model_name, seed=seed)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        acc = float(accuracy_score(y_te, pred))
        mf1 = macro_f1(y_te, pred)

        labels_sorted = sorted(pd.Series(pd.concat([y_tr, y_te])).unique().tolist())
        cm = confusion_matrix(y_te, pred, labels=labels_sorted)

        base = f"{task}__{model_name}"
        pd.DataFrame(cm, index=[f"true_{l}" for l in labels_sorted], columns=[f"pred_{l}" for l in labels_sorted]) \
            .to_csv(os.path.join(outdir, f"{base}__confusion.csv"))

        rep = classification_report(y_te, pred, digits=4, zero_division=0)
        write_text(os.path.join(outdir, f"{base}__report.txt"), rep)

        model_path = None
        if model_name in ("logreg", "rf"):
            model_path = os.path.join(outdir, f"{base}__model.joblib")
            joblib.dump(pipe, model_path)

        rows.append({
            "task": task,
            "setting": "both",
            "model": model_name,
            "macro_f1": mf1,
            "acc": acc,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "model_path": model_path,
        })

    return rows


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, default="fourthorse_trial_dummy_dataset.zip")
    parser.add_argument("--extract_dir", type=str, default="dataset_unzipped_upgrade")
    parser.add_argument("--outdir", type=str, default="poster_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    eval_dir = os.path.join(args.outdir, "_eval_artifacts")
    ensure_dir(eval_dir)

    # Load dataset from zip OR fallback to data/
    if os.path.exists(args.zip_path):
        paths = unzip_dataset(args.zip_path, args.extract_dir)
        steps = pd.read_csv(paths.steps_csv)
        choices = pd.read_csv(paths.choices_csv)
        typing = pd.read_csv(paths.typing_csv)
        dataset_hash = sha256_file(args.zip_path)
    else:
        steps = pd.read_csv("data/steps.csv") if os.path.exists("data/steps.csv") else pd.read_csv("steps.csv")
        choices = pd.read_csv("data/choices.csv") if os.path.exists("data/choices.csv") else pd.read_csv("choices.csv")
        typing = pd.read_csv("data/typing.csv") if os.path.exists("data/typing.csv") else pd.read_csv("typing.csv")
        dataset_hash = sha256_str("|".join([sha256_str(steps.to_csv(index=False)),
                                            sha256_str(choices.to_csv(index=False)),
                                            sha256_str(typing.to_csv(index=False))]))

    # Validate minimal schema
    validate_required_columns(steps, ["session_id", "latent_state"], "steps.csv")
    validate_required_columns(choices, ["session_id", "choice_point", "chosen_label"], "choices.csv")
    validate_required_columns(typing, ["session_id", "load_label"], "typing.csv")

    branch = choices[choices["choice_point"] == "branch"].copy()
    if len(branch) == 0:
        raise ValueError("No branch rows found in choices.csv (choice_point == 'branch').")

    # Feature columns (safe defaults)
    steps_features = [c for c in steps.columns if c not in ("latent_state",)]
    typing_features = [c for c in typing.columns if c not in ("load_label",)]

    # choices features: use only pre-choice features if they exist, else use numeric cols
    wanted = [
        "time_to_choice_ms",
        "pre_choice_backtrack_rate",
        "pre_choice_keys_per_sec",
        "pre_choice_mean_pause_ms",
        "pre_choice_terminal_activity",
    ]
    choices_features = [c for c in wanted if c in branch.columns]
    if len(choices_features) == 0:
        # fallback: all numeric except label/group
        choices_features = [c for c in branch.columns if c not in ("session_id", "chosen_label", "choice_point") and pd.api.types.is_numeric_dtype(branch[c])]

    metrics_rows = []
    metrics_rows += eval_task("latent_state", steps, "latent_state", steps_features, eval_dir, args.seed)
    metrics_rows += eval_task("chosen_label_branch", branch, "chosen_label", choices_features, eval_dir, args.seed)
    metrics_rows += eval_task("load_label", typing, "load_label", typing_features, eval_dir, args.seed)

    # Save metrics summary
    ms = pd.DataFrame(metrics_rows)[["model", "setting", "task", "macro_f1", "acc", "n_train", "n_test"]]
    ms.to_csv(os.path.join(args.outdir, "metrics_summary.csv"), index=False)

    # Commitments + privacy spec (Stage 1)
    # feature schema hash: just for latent_state features used (excluding session_id)
    schema = sorted([c for c in steps_features if c != "session_id" and c != "latent_state"])
    feature_schema_hash = sha256_str("|".join(schema))

    # model hashes (hash joblib files we wrote)
    model_hashes = {}
    for fn in os.listdir(eval_dir):
        if fn.endswith(".joblib"):
            model_hashes[fn] = sha256_file(os.path.join(eval_dir, fn))

    commitments = {
        "dataset_hash": dataset_hash,
        "feature_schema_hash": feature_schema_hash,
        "policy_hash": sha256_str("stage1_no_bandit_yet"),
        "model_hashes": model_hashes,
    }
    write_json(os.path.join(args.outdir, "commitments.json"), commitments)

    privacy_spec = "\n".join([
        "- No raw text collected.",
        "- No raw keystroke event streams stored.",
        "- Only aggregated features per step/episode are used (dwell/backtrack/typing rates/pause stats/etc.).",
        "- session_id is used only for group train/test splitting and removed from model features.",
        "- Models are versioned by SHA-256 hashes for auditability.",
    ])
    write_text(os.path.join(args.outdir, "privacy_spec.txt"), privacy_spec)

    print("DONE. Wrote poster_outputs/ with metrics_summary.csv + commitments.json + privacy_spec.txt")


if __name__ == "__main__":
    main()

