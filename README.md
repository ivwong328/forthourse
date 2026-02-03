# forthourse
"""
Fourthorse Trial Mode — End-to-End Baseline ML Pipeline

What this does:
1) Unzips fourthorse_trial_dummy_dataset.zip (steps/choices/typing CSVs)
2) Validates required columns
3) Profiles dataset + label distributions
4) Trains + evaluates 3 baseline models with GROUP split by session_id:
   - steps.csv   -> latent_state (confused/engaged/confident)
   - choices.csv (branch only) -> chosen_label (P/D/B/H)
   - typing.csv  -> load_label (low_load/high_load)
5) Saves: reports (.txt/.json), confusion matrices (.csv), trained models (.joblib)

Run:
  python fourthorse_pipeline.py --zip_path fourthorse_trial_dummy_dataset.zip --outdir results

Notes:
- Uses ONLY pre-choice features for strategy choice (filters choices where choice_point == "branch").
- Splits by session_id to prevent leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.inspection import permutation_importance

import joblib


# -----------------------------
# Config + utilities
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


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def df_label_distribution(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    return df[label_col].value_counts(dropna=False).to_dict()


def validate_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


def group_split_idx(
    df: pd.DataFrame,
    label_col: str,
    group_col: str = "session_id",
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    if group_col not in df.columns:
        raise ValueError(f"Expected group column '{group_col}' in dataframe.")

    y = df[label_col]
    X = df.drop(columns=[label_col])

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=df[group_col]))
    return train_idx, test_idx

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocess numeric + categorical columns safely.
    """
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop"
    )
    return pre


def build_model_pipeline(X: pd.DataFrame, class_weight: Optional[str] = "balanced") -> Pipeline:
    """
    Baseline model: Logistic Regression with preprocessing.
    """
    pre = build_preprocessor(X)
    clf = LogisticRegression(
        max_iter=5000,
        class_weight=class_weight,
        n_jobs=None
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def save_confusion_matrix(cm: np.ndarray, labels: List[str], out_csv: str) -> None:
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(out_csv, index=True)


def safe_feature_names(pipe: Pipeline) -> List[str]:
    """
    Try to extract feature names after preprocessing.
    Works for OneHotEncoder.
    """
    pre: ColumnTransformer = pipe.named_steps["pre"]
    names: List[str] = []

    for name, transformer, cols in pre.transformers_:
        if name == "remainder":
            continue

        if hasattr(transformer, "named_steps"):
            # Pipeline
            last = list(transformer.named_steps.values())[-1]
        else:
            last = transformer

        if isinstance(last, OneHotEncoder):
            try:
                ohe_names = last.get_feature_names_out(cols)
                names.extend(ohe_names.tolist())
            except Exception:
                # fallback if encoder isn't fitted or cols mismatch
                names.extend([f"{c}__ohe" for c in cols])
        else:
            names.extend([str(c) for c in cols])

    return names


def run_task(
    df: pd.DataFrame,
    task_name: str,
    label_col: str,
    drop_cols: List[str],
    outdir: str,
    seed: int = 42
) -> Dict[str, object]:
    """
    Train + evaluate a baseline classifier for one task.
    Saves:
      - report.txt, report.json
      - confusion_matrix.csv
      - model.joblib
      - permutation_importance_top.csv
    """
    ensure_dir(outdir)

    # Basic cleanup
    df2 = df.copy()
    df2 = df2.dropna(subset=[label_col])

    # Drop explicitly requested columns
    for c in drop_cols:
        if c in df2.columns:
            df2 = df2.drop(columns=[c])

    # Split
    X_train, X_test, y_train, y_test = group_split(df2, label_col=label_col, seed=seed)

    # Train
    pipe = build_model_pipeline(X_train, class_weight="balanced")
    pipe.fit(X_train, y_train)

    # Predict
    preds = pipe.predict(X_test)

    # Metrics
    labels_sorted = sorted(list(pd.Series(y_test).dropna().unique()))
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    acc = accuracy_score(y_test, preds)
    report_txt = classification_report(y_test, preds, digits=4)
    report_obj = {
        "task": task_name,
        "label_col": label_col,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": float(acc),
        "labels": labels_sorted,
        "classification_report": report_txt
    }

    # Save outputs
    write_text(os.path.join(outdir, "report.txt"), f"{task_name}\n\nAccuracy: {acc:.4f}\n\n{report_txt}\n")
    write_json(os.path.join(outdir, "report.json"), report_obj)
    save_confusion_matrix(cm, labels_sorted, os.path.join(outdir, "confusion_matrix.csv"))
    joblib.dump(pipe, os.path.join(outdir, "model.joblib"))

    # Permutation importance (optional but useful)
    # NOTE: This runs on the pipeline; sklearn will permute raw input columns.
    try:
        r = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=5,
            random_state=seed,
            scoring="f1_macro"
        )
        importances = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std
        }).sort_values("importance_mean", ascending=False)

        importances.head(25).to_csv(os.path.join(outdir, "permutation_importance_top25.csv"), index=False)
    except Exception as e:
        write_text(os.path.join(outdir, "permutation_importance_error.txt"), str(e))

    return {
        "task": task_name,
        "accuracy": acc,
        "labels": labels_sorted
    }


def profile_data(steps: pd.DataFrame, choices: pd.DataFrame, typing: pd.DataFrame) -> Dict[str, object]:
    """
    Basic dataset profiling and label distributions.
    """
    prof = {
        "steps": {
            "shape": [int(steps.shape[0]), int(steps.shape[1])],
            "unique_sessions": int(steps["session_id"].nunique()) if "session_id" in steps.columns else None,
            "label_dist_latent_state": df_label_distribution(steps, "latent_state") if "latent_state" in steps.columns else None
        },
        "choices": {
            "shape": [int(choices.shape[0]), int(choices.shape[1])],
            "unique_sessions": int(choices["session_id"].nunique()) if "session_id" in choices.columns else None,
            "choice_point_dist": df_label_distribution(choices, "choice_point") if "choice_point" in choices.columns else None,
        },
        "typing": {
            "shape": [int(typing.shape[0]), int(typing.shape[1])],
            "unique_sessions": int(typing["session_id"].nunique()) if "session_id" in typing.columns else None,
            "label_dist_load_label": df_label_distribution(typing, "load_label") if "load_label" in typing.columns else None
        }
    }
    return prof


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fourthorse Trial Mode ML Pipeline")
    parser.add_argument("--zip_path", type=str, default="fourthorse_trial_dummy_dataset.zip",
                        help="Path to fourthorse_trial_dummy_dataset.zip")
    parser.add_argument("--extract_dir", type=str, default="dataset_unzipped",
                        help="Where to unzip the dataset")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory for reports/models")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    zip_path = args.zip_path
    if not os.path.exists(zip_path):
        # try relative to current working directory
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    ensure_dir(args.outdir)

    # 1) Unzip
    paths = unzip_dataset(zip_path, args.extract_dir)

    # 2) Load
    steps = load_csv(paths.steps_csv)
    choices = load_csv(paths.choices_csv)
    typing = load_csv(paths.typing_csv)

    # 3) Validate schemas (minimal required columns)
    validate_required_columns(steps, ["session_id", "latent_state"], "steps.csv")
    validate_required_columns(choices, ["session_id", "choice_point", "chosen_label"], "choices.csv")
    validate_required_columns(typing, ["session_id", "load_label"], "typing.csv")

    # 4) Profile
    prof = profile_data(steps, choices, typing)
    write_json(os.path.join(args.outdir, "data_profile.json"), prof)

    # 5) Task A — latent state inference (steps.csv)
    # We DROP session_id to avoid memorization; keep screen_id if present (it’s legit context).
    taskA_dir = os.path.join(args.outdir, "task_latent_state_steps")
    taskA = run_task(
        df=steps,
        task_name="Latent State (steps.csv)",
        label_col="latent_state",
        drop_cols=["session_id"],  # keep step_idx, screen_id, etc.
        outdir=taskA_dir,
        seed=args.seed
    )

    # 6) Task B — strategy choice prediction (branch-only rows)
    # IMPORTANT: filter to choice_point == "branch" so chosen_label is P/D/B/H.
    branch = choices.copy()
    if "choice_point" in branch.columns:
        branch = branch[branch["choice_point"] == "branch"].copy()

    # If someone accidentally passes a dataset without branch rows, fail loudly.
    if len(branch) == 0:
        raise ValueError("No branch choice rows found (choice_point == 'branch').")

    taskB_dir = os.path.join(args.outdir, "task_strategy_choice_branch")
    taskB = run_task(
        df=branch,
        task_name="Strategy Choice (branch rows in choices.csv)",
        label_col="chosen_label",
        drop_cols=["session_id", "choice_point"],
        outdir=taskB_dir,
        seed=args.seed
    )

    # 7) Task C — typing load classification (typing.csv)
    taskC_dir = os.path.join(args.outdir, "task_typing_load")
    taskC = run_task(
        df=typing,
        task_name="Typing Load (typing.csv)",
        label_col="load_label",
        drop_cols=["session_id"],
        outdir=taskC_dir,
        seed=args.seed
    )

    # 8) Summary
    summary = {
        "task_latent_state_steps": {"accuracy": float(taskA["accuracy"]), "labels": taskA["labels"]},
        "task_strategy_choice_branch": {"accuracy": float(taskB["accuracy"]), "labels": taskB["labels"]},
        "task_typing_load": {"accuracy": float(taskC["accuracy"]), "labels": taskC["labels"]},
        "outputs_dir": os.path.abspath(args.outdir),
    }
    write_json(os.path.join(args.outdir, "summary.json"), summary)

    print("\n=== DONE ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
