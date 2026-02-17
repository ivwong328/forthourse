from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
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
# Config
# -----------------------------

TYPING_KW = ["pause", "backspace", "keys", "kps", "cpm", "wpm", "burst", "interkey", "rewrite", "chars", "paste"]
NAV_KW = ["dwell", "latency", "nav", "backtrack", "scroll", "step_time", "time_on", "time_since", "step_idx"]

PERSONAS = ["NEUTRAL", "SOCRATIC", "COMPRESSION", "COUNTER"]


# -----------------------------
# Utilities
# -----------------------------

@dataclass
class DatasetPaths:
    steps_csv: str
    choices_csv: str
    typing_csv: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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


def validate_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


def infer_feature_groups(cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    nav, typing, other = [], [], []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in TYPING_KW):
            typing.append(c)
        elif any(k in cl for k in NAV_KW):
            nav.append(c)
        else:
            other.append(c)
    return nav, typing, other


# -----------------------------
# Split (group if possible; fallback if only 1 group)
# -----------------------------

def group_split_idx(
    df: pd.DataFrame,
    label_col: str,
    group_col: str = "session_id",
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    # If not enough groups, do a simple shuffled split by rows
    if group_col not in df.columns:
        raise ValueError(f"Missing group col: {group_col}")

    n_groups = df[group_col].nunique(dropna=False)
    if n_groups < 2:
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        split = max(1, int(len(idx) * (1 - test_size)))
        # ensure at least 1 test row if possible
        if split >= len(idx):
            split = len(idx) - 1
        if split < 1:
            split = 1
        return idx[:split], idx[split:]

    y = df[label_col]
    X = df.drop(columns=[label_col])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups=df[group_col]))
    return train_idx, test_idx


# -----------------------------
# Preprocess + models
# -----------------------------

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


def build_model(name: str, seed: int = 42):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(max_iter=5000, class_weight="balanced")
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(X: pd.DataFrame, model_name: str, seed: int = 42) -> Pipeline:
    return Pipeline(steps=[
        ("pre", build_preprocessor(X)),
        ("clf", build_model(model_name, seed=seed)),
    ])


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# -----------------------------
# Task evaluation
# -----------------------------

def eval_task(
    task: str,
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    eval_dir: str,
    seed: int,
    setting: str = "both",
) -> List[Dict[str, object]]:
    ensure_dir(eval_dir)

    needed = ["session_id", label_col] + [c for c in feature_cols if c in df.columns]
    df2 = df[needed].dropna(subset=[label_col]).copy()

    tr_idx, te_idx = group_split_idx(df2, label_col=label_col, group_col="session_id", seed=seed)
    tr = df2.iloc[tr_idx].copy()
    te = df2.iloc[te_idx].copy()

    y_tr = tr[label_col].astype(str)
    y_te = te[label_col].astype(str)
    X_tr = tr.drop(columns=[label_col, "session_id"])
    X_te = te.drop(columns=[label_col, "session_id"])

    labels_sorted = sorted(pd.Series(pd.concat([y_tr, y_te])).unique().tolist())

    rows: List[Dict[str, object]] = []
    for model_name in ["majority", "logreg", "rf"]:
        pipe = build_pipeline(X_tr, model_name=model_name, seed=seed)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        acc = float(accuracy_score(y_te, pred))
        mf1 = macro_f1(y_te, pred)

        cm = confusion_matrix(y_te, pred, labels=labels_sorted)
        cm_path = os.path.join(eval_dir, f"{task}__{setting}__{model_name}__confusion.csv")
        pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_sorted],
            columns=[f"pred_{l}" for l in labels_sorted],
        ).to_csv(cm_path, index=True)

        rep = classification_report(y_te, pred, digits=4, zero_division=0)
        write_text(os.path.join(eval_dir, f"{task}__{setting}__{model_name}__report.txt"), rep)

        model_path = None
        if model_name in ("logreg", "rf"):
            model_path = os.path.join(eval_dir, f"{task}__{setting}__{model_name}.joblib")
            joblib.dump(pipe, model_path)

        rows.append({
            "task": task,
            "setting": setting,
            "model": model_name,
            "macro_f1": mf1,
            "acc": acc,
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "model_path": model_path,
        })

    return rows


# -----------------------------
# Trajectory (one session)
# -----------------------------

def make_trajectory_plot(
    outdir: str,
    fig_dir: str,
    latent_pipe: Pipeline,
    steps_df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
) -> None:
    # Pick one "session" (if only one exists, still fine)
    sessions = steps_df["session_id"].astype(str).unique().tolist()
    if not sessions:
        return

    example_sid = sessions[0]
    ex = steps_df[steps_df["session_id"].astype(str) == example_sid].copy()

    # Sorting if we have step index
    for cand in ["step_idx", "step_index", "idx", "t", "time"]:
        if cand in ex.columns:
            ex = ex.sort_values(cand)
            break

    X_ex = ex[feature_cols].copy()
    probs = latent_pipe.predict_proba(X_ex)
    classes = latent_pipe.named_steps["clf"].classes_.tolist()

    if "step_idx" in ex.columns:
        step_axis = ex["step_idx"].values
    elif "step_index" in ex.columns:
        step_axis = ex["step_index"].values
    else:
        step_axis = np.arange(1, len(ex) + 1)

    traj = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
    traj.insert(0, "step_idx", step_axis)
    traj.insert(0, "session_id", example_sid)
    traj_path = os.path.join(outdir, "trajectory_example.csv")
    traj.to_csv(traj_path, index=False)

    plt.figure()
    for c in classes:
        plt.plot(traj["step_idx"], traj[f"p_{c}"], label=c)
    plt.xlabel("Step")
    plt.ylabel("Predicted probability")
    plt.title("State Trajectory Example")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_state_trajectory_example.png"), dpi=200)
    plt.close()


# -----------------------------
# Bandit (Thompson Sampling)
# -----------------------------

def thompson_pick(alpha: Dict[str, float], beta: Dict[str, float], rng: np.random.Generator) -> str:
    samples = {p: rng.beta(alpha[p], beta[p]) for p in PERSONAS}
    return max(samples, key=samples.get)


def run_bandit(
    outdir: str,
    fig_dir: str,
    policy: dict,
    p_confused: np.ndarray,
    seed: int,
) -> str:
    rng = np.random.default_rng(seed)
    alpha = {p: float(policy["bandit"]["alpha0"]) for p in PERSONAS}
    beta = {p: float(policy["bandit"]["beta0"]) for p in PERSONAS}
    counts = {p: 0 for p in PERSONAS}

    rows = []
    running = 0.0

    episodes = int(policy["episodes"])
    for i in range(episodes):
        pc = float(p_confused[i % len(p_confused)])
        persona = thompson_pick(alpha, beta, rng)
        counts[persona] += 1

        # base reward
        r = 1.0 - pc

        # persona effects
        if pc > 0.6 and persona == "COMPRESSION":
            r += 0.10
        if pc < 0.4 and persona == "SOCRATIC":
            r += 0.05
        if pc > 0.6 and persona == "COUNTER":
            r -= 0.05

        r = float(np.clip(r, 0.0, 1.0))
        running = (running * i + r) / (i + 1)

        # Beta update using success threshold
        success = 1 if r >= 0.5 else 0
        alpha[persona] += success
        beta[persona] += (1 - success)

        total = sum(counts.values())
        prob_row = {f"prob_{p.lower()}": counts[p] / total for p in PERSONAS}

        rows.append({
            "episode": i + 1,
            "chosen_persona": persona,
            "p_confused": pc,
            "reward": r,
            "running_avg_reward": running,
            **prob_row,
        })

    bandit_df = pd.DataFrame(rows)
    bandit_df.to_csv(os.path.join(outdir, "bandit_learning_curve.csv"), index=False)

    plt.figure()
    plt.plot(bandit_df["episode"], bandit_df["running_avg_reward"])
    plt.xlabel("Episode")
    plt.ylabel("Running Avg Reward")
    plt.title("Bandit Convergence")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_bandit_convergence.png"), dpi=200)
    plt.close()

    policy_hash = sha256_str(json.dumps(policy, sort_keys=True))
    return policy_hash


# -----------------------------
# Plots
# -----------------------------

def plot_model_comparison_latent(metrics_df: pd.DataFrame, fig_dir: str) -> None:
    df = metrics_df[(metrics_df["task"] == "latent_state") & (metrics_df["setting"] == "both")].copy()
    if df.empty:
        return

    order = ["majority", "logreg", "rf"]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")

    plt.figure()
    plt.bar(df["model"].astype(str), df["macro_f1"].astype(float))
    plt.ylabel("Macro-F1")
    plt.title("Model Comparison — Latent State")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_model_comparison_latent_state.png"), dpi=200)
    plt.close()


def plot_ablation(ablation_df: pd.DataFrame, fig_dir: str) -> None:
    # Plot only logreg rows for the ablation bar chart
    df = ablation_df[ablation_df["model"] == "logreg"].copy()
    if df.empty:
        return

    order = ["nav_only", "typing_only", "both", "both_without_screen_id"]
    df["setting"] = pd.Categorical(df["setting"], categories=order, ordered=True)
    df = df.sort_values("setting")

    plt.figure()
    plt.bar(df["setting"].astype(str), df["macro_f1"].astype(float))
    plt.ylabel("Macro-F1")
    plt.title("Ablation — Latent State (LogReg)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_ablation_latent_state.png"), dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fourthorse Upgrade Pipeline (poster outputs)")
    parser.add_argument("--zip_path", type=str, default="fourthorse_trial_dummy_dataset.zip")
    parser.add_argument("--extract_dir", type=str, default="dataset_unzipped_upgrade")
    parser.add_argument("--outdir", type=str, default="poster_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    eval_dir = os.path.join(args.outdir, "_eval_artifacts")
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dir(eval_dir)
    ensure_dir(fig_dir)

    # -------------------------
    # Load data: zip OR fallback
    # -------------------------
    dataset_hash = None
    if os.path.exists(args.zip_path):
        paths = unzip_dataset(args.zip_path, args.extract_dir)
        steps = pd.read_csv(paths.steps_csv)
        choices = pd.read_csv(paths.choices_csv)
        typing = pd.read_csv(paths.typing_csv)
        dataset_hash = sha256_file(args.zip_path)
    else:
        # fallback to repo data/
        steps = pd.read_csv("data/steps.csv") if os.path.exists("data/steps.csv") else pd.read_csv("steps.csv")
        choices = pd.read_csv("data/choices.csv") if os.path.exists("data/choices.csv") else pd.read_csv("choices.csv")
        typing = pd.read_csv("data/typing.csv") if os.path.exists("data/typing.csv") else pd.read_csv("typing.csv")
        # stable-ish "dataset hash" from file contents
        dataset_hash = sha256_str("|".join([
            sha256_str(steps.to_csv(index=False)),
            sha256_str(choices.to_csv(index=False)),
            sha256_str(typing.to_csv(index=False)),
        ]))

    validate_required_columns(steps, ["session_id", "latent_state"], "steps.csv")
    validate_required_columns(choices, ["session_id", "choice_point", "chosen_label"], "choices.csv")
    validate_required_columns(typing, ["session_id", "load_label"], "typing.csv")

    # branch-only
    branch = choices[choices["choice_point"] == "branch"].copy()
    if len(branch) == 0:
        raise ValueError("No branch rows found (choice_point == 'branch').")

    # -------------------------
    # Feature lists
    # -------------------------
    steps_all_feats = [c for c in steps.columns if c not in ["latent_state", "session_id"]]
    nav_cols, typing_cols, other_cols = infer_feature_groups(steps_all_feats)
    both_cols = sorted(set(nav_cols + typing_cols + other_cols))

    ablation_settings = {
        "nav_only": sorted(set(nav_cols + other_cols)),
        "typing_only": sorted(set(typing_cols + other_cols)),
        "both": both_cols,
        "both_without_screen_id": [c for c in both_cols if c != "screen_id"],
    }
    # If screen_id doesn't exist, remove that setting to avoid duplicates
    if "screen_id" not in steps.columns:
        ablation_settings.pop("both_without_screen_id", None)

    typing_features = [c for c in typing.columns if c not in ["load_label", "session_id"]]

    wanted = [
        "time_to_choice_ms",
        "pre_choice_backtrack_rate",
        "pre_choice_keys_per_sec",
        "pre_choice_mean_pause_ms",
        "pre_choice_terminal_activity",
    ]
    choices_features = [c for c in wanted if c in branch.columns]
    if len(choices_features) == 0:
        choices_features = [
            c for c in branch.columns
            if c not in ["session_id", "chosen_label", "choice_point"]
            and pd.api.types.is_numeric_dtype(branch[c])
        ]

    # -------------------------
    # 1) Model comparisons (all tasks)
    # -------------------------
    metrics_rows: List[Dict[str, object]] = []
    metrics_rows += eval_task("latent_state", steps, "latent_state", both_cols, eval_dir, args.seed, setting="both")
    metrics_rows += eval_task("chosen_label_branch", branch, "chosen_label", choices_features, eval_dir, args.seed, setting="both")
    metrics_rows += eval_task("load_label", typing, "load_label", typing_features, eval_dir, args.seed, setting="both")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.outdir, "metrics_summary.csv"), index=False)

    # Figure: model comparison (latent_state)
    plot_model_comparison_latent(metrics_df, fig_dir)

    # -------------------------
    # 2) Ablation study (latent_state)
    # -------------------------
    ablation_rows: List[Dict[str, object]] = []
    for setting, feats in ablation_settings.items():
        rows = eval_task("latent_state", steps, "latent_state", feats, eval_dir, args.seed, setting=setting)
        # keep only logreg/rf in ablation results (spec allows)
        for r in rows:
            if r["model"] in ["logreg", "rf"]:
                ablation_rows.append({
                    "task": "latent_state",
                    "setting": setting,
                    "model": r["model"],
                    "macro_f1": r["macro_f1"],
                    "acc": r["acc"],
                    "n_train": r["n_train"],
                    "n_test": r["n_test"],
                })

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(os.path.join(args.outdir, "ablation_results.csv"), index=False)
    plot_ablation(ablation_df, fig_dir)

    # -------------------------
    # 3) Train final latent logreg (for trajectory + bandit)
    # -------------------------
    # Split once
    steps_for_final = steps[["session_id", "latent_state"] + both_cols].dropna(subset=["latent_state"]).copy()
    tr_idx, te_idx = group_split_idx(steps_for_final, label_col="latent_state", group_col="session_id", seed=args.seed)
    train_df = steps_for_final.iloc[tr_idx].copy()
    test_df = steps_for_final.iloc[te_idx].copy()

    X_tr = train_df[both_cols].copy()
    y_tr = train_df["latent_state"].astype(str)
    X_te = test_df[both_cols].copy()
    y_te = test_df["latent_state"].astype(str)

    latent_pipe = build_pipeline(X_tr, model_name="logreg", seed=args.seed)
    latent_pipe.fit(X_tr, y_tr)
    joblib.dump(latent_pipe, os.path.join(eval_dir, "latent_state__final_logreg.joblib"))

    # -------------------------
    # 4) Trajectory artifact (required)
    # -------------------------
    # For trajectory, pass a df that includes session_id and feature cols
    steps_test_for_traj = test_df[["session_id"] + both_cols].copy()
    make_trajectory_plot(args.outdir, fig_dir, latent_pipe, pd.concat([test_df[["session_id"] + both_cols]], axis=0), both_cols, args.seed)

    # -------------------------
    # 5) Bandit loop (required)
    # -------------------------
    # compute p_confused for test rows
    probs = latent_pipe.predict_proba(X_te)
    cls = latent_pipe.named_steps["clf"].classes_.tolist()
    conf_idx = cls.index("confused") if "confused" in cls else 0
    p_confused = probs[:, conf_idx]
    if len(p_confused) == 0:
        # very tiny dataset fallback
        p_confused = np.array([0.5], dtype=float)

    policy = {
        "personas": PERSONAS,
        "bandit": {"type": "thompson_beta_bernoulli", "alpha0": 1.0, "beta0": 1.0},
        "reward": {
            "base": "1 - p_confused",
            "rules": [
                {"if": "p_confused>0.6", "persona": "COMPRESSION", "delta": 0.10},
                {"if": "p_confused<0.4", "persona": "SOCRATIC", "delta": 0.05},
                {"if": "p_confused>0.6", "persona": "COUNTER", "delta": -0.05},
            ],
            "clamp": [0.0, 1.0],
        },
        "episodes": int(args.episodes),
        "seed": int(args.seed),
    }
    write_json(os.path.join(args.outdir, "persona_policy.json"), policy)

    policy_hash = run_bandit(args.outdir, fig_dir, policy, p_confused, args.seed)

    # -------------------------
    # 6) Integration artifacts (required)
    # -------------------------
    # feature_schema.json
    schema = []
    for c in both_cols:
        dtype = str(steps[c].dtype) if c in steps.columns else "unknown"
        kind = "numeric" if (c in steps.columns and pd.api.types.is_numeric_dtype(steps[c])) else "categorical"
        schema.append({"name": c, "dtype": dtype, "kind": kind})
    write_json(os.path.join(args.outdir, "feature_schema.json"), schema)

    # label_map.json
    label_map = {lbl: i for i, lbl in enumerate(cls)}
    write_json(os.path.join(args.outdir, "label_map.json"), label_map)

    # -------------------------
    # 7) Commitments + privacy
    # -------------------------
    # feature_schema_hash
    feature_schema_hash = sha256_str("|".join(sorted(both_cols)))

    # model_hashes
    model_hashes = {}
    for fn in os.listdir(eval_dir):
        if fn.endswith(".joblib"):
            model_hashes[fn] = sha256_file(os.path.join(eval_dir, fn))

    commitments = {
        "dataset_hash": dataset_hash,
        "feature_schema_hash": feature_schema_hash,
        "policy_hash": policy_hash,
        "model_hashes": model_hashes,
    }
    write_json(os.path.join(args.outdir, "commitments.json"), commitments)

    privacy_spec = "\n".join([
        "- No raw text collected.",
        "- No raw keystroke event streams stored.",
        "- Only aggregated features per step/episode are used (dwell/backtrack/typing rates/pause stats/etc.).",
        "- session_id is used only for group train/test splitting; removed from model features.",
        "- Models/policies are versioned by SHA-256 hashes to support auditability.",
    ])
    write_text(os.path.join(args.outdir, "privacy_spec.txt"), privacy_spec)

    print("DONE")


if __name__ == "__main__":
    main()
