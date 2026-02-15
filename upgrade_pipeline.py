cat > upgrade_pipeline.py << 'EOF'
from __future__ import annotations

import argparse
import json
import os
import zipfile
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier


# -----------------------------
# Paths / IO
# -----------------------------

@dataclass
class DatasetPaths:
    steps_csv: str
    choices_csv: str
    typing_csv: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


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
    return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


# -----------------------------
# Privacy layer
# -----------------------------

SENSITIVE_PATTERNS = [
    "text", "raw", "keystroke", "key_event", "event_stream", "payload", "message", "prompt"
]

def sanitize(df: pd.DataFrame, allowed_cols: List[str]) -> pd.DataFrame:
    """
    Feature-only logging: drop anything not explicitly allowed.
    Also drops any column that matches sensitive patterns, even if mistakenly allowed.
    """
    keep = [c for c in allowed_cols if c in df.columns]
    out = df[keep].copy()

    # hard drop sensitive columns if present
    drop = []
    for c in out.columns:
        lc = c.lower()
        if any(p in lc for p in SENSITIVE_PATTERNS):
            drop.append(c)
    if drop:
        out = out.drop(columns=drop)
    return out


def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_str(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


# -----------------------------
# Splitting / preprocessing
# -----------------------------

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
            n_estimators=400,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
            min_samples_leaf=2,
        )
    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(X: pd.DataFrame, model_name: str, seed: int = 42) -> Pipeline:
    pre = build_preprocessor(X)
    clf = build_model(model_name, seed=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


# -----------------------------
# Evaluation helpers
# -----------------------------

def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def save_confusion_matrix_csv(cm: np.ndarray, labels: List[str], out_csv: str) -> None:
    df_cm = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    df_cm.to_csv(out_csv, index=True)


def run_one_model(
    task: str,
    setting: str,
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    group_col: str,
    model_name: str,
    outdir: str,
    seed: int
) -> Dict[str, object]:
    ensure_dir(outdir)

    # feature-only + sanitize
    allowed = [group_col, label_col] + feature_cols
    df2 = sanitize(df, allowed_cols=allowed).dropna(subset=[label_col])

    # split by session_id
    tr_idx, te_idx = group_split_idx(df2, label_col=label_col, group_col=group_col, seed=seed)
    tr = df2.iloc[tr_idx].copy()
    te = df2.iloc[te_idx].copy()

    y_train = tr[label_col].astype(str)
    y_test  = te[label_col].astype(str)

    X_train = tr.drop(columns=[label_col])
    X_test  = te.drop(columns=[label_col])

    # drop group from model features always
    if group_col in X_train.columns:
        X_train = X_train.drop(columns=[group_col])
    if group_col in X_test.columns:
        X_test = X_test.drop(columns=[group_col])

    pipe = build_pipeline(X_train, model_name=model_name, seed=seed)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    labels_sorted = sorted(pd.Series(pd.concat([y_train, y_test])).unique().tolist())
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    acc = float(accuracy_score(y_test, preds))
    mf1 = macro_f1(y_test, preds)
    rep_txt = classification_report(y_test, preds, digits=4, zero_division=0)

    # save artifacts per model/task/setting
    base = f"{task}__{setting}__{model_name}"
    write_text(os.path.join(outdir, f"{base}__report.txt"), rep_txt)
    save_confusion_matrix_csv(cm, labels_sorted, os.path.join(outdir, f"{base}__confusion.csv"))

    # permutation importance for logreg/rf only (skip majority)
    feat_imp_path = os.path.join(outdir, f"{base}__perm_importance_top25.csv")
    if model_name in ("logreg", "rf"):
        try:
            r = permutation_importance(pipe, X_test, y_test, n_repeats=5, random_state=seed, scoring="f1_macro")
            imp = pd.DataFrame({
                "feature": X_test.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std,
            }).sort_values("importance_mean", ascending=False)
            imp.head(25).to_csv(feat_imp_path, index=False)
        except Exception as e:
            write_text(os.path.join(outdir, f"{base}__perm_importance_error.txt"), str(e))

    # save model for later use (only for logreg/rf)
    model_path = None
    if model_name in ("logreg", "rf") and task == "latent_state" and setting == "both":
        model_path = os.path.join(outdir, f"{base}__model.joblib")
        joblib.dump(pipe, model_path)

    return {
        "task": task,
        "setting": setting,
        "model": model_name,
        "macro_f1": mf1,
        "acc": acc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_path": model_path,
    }


# -----------------------------
# Feature groups for ablation
# -----------------------------

NAV_KW = ["dwell", "backtrack", "nav", "step", "time", "scroll"]
TYP_KW = ["pause", "backspace", "keys", "cpm", "kps", "burst", "rewrite", "hesitation"]

def infer_feature_groups(step_feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    nav = []
    typ = []
    for c in step_feature_cols:
        lc = c.lower()
        if any(k in lc for k in NAV_KW):
            nav.append(c)
        if any(k in lc for k in TYP_KW):
            typ.append(c)
    # avoid empty groups if naming differs
    if len(nav) == 0:
        nav = [c for c in step_feature_cols if "nav" in c.lower() or "dwell" in c.lower()]
    if len(typ) == 0:
        typ = [c for c in step_feature_cols if "pause" in c.lower() or "keys" in c.lower() or "backspace" in c.lower()]
    return nav, typ


# -----------------------------
# Plotting
# -----------------------------

def save_bar_chart(out_png: str, title: str, labels: List[str], values: List[float], ylabel: str):
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_trajectory_plot(out_png: str, session_df: pd.DataFrame, proba: np.ndarray, class_names: List[str]):
    # proba: [n_steps, n_classes]
    x = session_df["step_idx"].values if "step_idx" in session_df.columns else np.arange(len(session_df))

    plt.figure(figsize=(9, 4.5))
    for i, cname in enumerate(class_names):
        plt.plot(x, proba[:, i], label=f"p({cname})")
    plt.title("Predicted latent-state probabilities over steps (example session)")
    plt.xlabel("step_idx")
    plt.ylabel("probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_bandit_plot(out_png: str, curve_df: pd.DataFrame, persona_cols: List[str]):
    plt.figure(figsize=(9, 4.5))
    x = curve_df["episode"].values
    for col in persona_cols:
        plt.plot(x, curve_df[col].values, label=col)
    plt.title("Bandit convergence (posterior mean persona probability)")
    plt.xlabel("episode")
    plt.ylabel("probability")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Bandit policy demo (Thompson Sampling)
# -----------------------------

PERSONAS = ["P0_Neutral", "P1_Socratic", "P2_Compression", "P3_CounterAgent"]

POLICY_CONFIG_STR = """
policy=thompson_sampling
personas=P0_Neutral,P1_Socratic,P2_Compression,P3_CounterAgent
reward=1 - 0.7*p_confused - 0.3*p_high_load + persona_effect
persona_effect:
  P2_Compression: +0.12 when confused/high_load
  P1_Socratic: +0.08 when engaged
  P0_Neutral: +0.00 baseline
  P3_CounterAgent: -0.10 when confused/high_load
"""

def persona_effect(persona: str, p_confused: float, p_engaged: float, p_confident: float, p_high_load: float) -> float:
    # small simulated causal effect to make the bandit learn something
    if persona == "P2_Compression":
        return 0.12 * (0.6 * p_confused + 0.4 * p_high_load)
    if persona == "P1_Socratic":
        return 0.08 * (p_engaged)
    if persona == "P3_CounterAgent":
        return -0.10 * (0.7 * p_confused + 0.3 * p_high_load)
    return 0.0


def thompson_bandit_sim(
    steps_df: pd.DataFrame,
    latent_pipe: Pipeline,
    latent_feature_cols: List[str],
    typing_df: pd.DataFrame,
    typing_pipe: Pipeline,
    typing_feature_cols: List[str],
    out_csv: str,
    out_png: str,
    seed: int = 42,
    episodes: int = 300
):
    rng = np.random.default_rng(seed)

    # Beta priors per persona
    alpha = {p: 1.0 for p in PERSONAS}
    beta = {p: 1.0 for p in PERSONAS}

    # prepare label order for latent_state probs
    latent_classes = list(latent_pipe.named_steps["clf"].classes_)

    # for typing probs
    typing_classes = list(typing_pipe.named_steps["clf"].classes_)

    def get_latent_probs(row_df: pd.DataFrame) -> Dict[str, float]:
        X = row_df[latent_feature_cols].copy()
        if "session_id" in X.columns:
            X = X.drop(columns=["session_id"])
        proba = latent_pipe.predict_proba(X)[0]
        return {latent_classes[i]: float(proba[i]) for i in range(len(latent_classes))}

    def get_typing_probs(row_df: pd.DataFrame) -> Dict[str, float]:
        X = row_df[typing_feature_cols].copy()
        if "session_id" in X.columns:
            X = X.drop(columns=["session_id"])
        proba = typing_pipe.predict_proba(X)[0]
        return {typing_classes[i]: float(proba[i]) for i in range(len(typing_classes))}

    curve_rows = []
    running = 0.0

    # pre-sanitize working frames (feature-only)
    latent_allowed = ["session_id"] + latent_feature_cols
    typing_allowed = ["session_id"] + typing_feature_cols

    steps_s = sanitize(steps_df, latent_allowed).dropna()
    typing_s = sanitize(typing_df, typing_allowed).dropna()

    for ep in range(1, episodes + 1):
        # sample one situation row from steps + one from typing (simulate environment snapshot)
        srow = steps_s.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).copy()
        trow = typing_s.sample(n=1, random_state=int(rng.integers(0, 1_000_000))).copy()

        lp = get_latent_probs(srow)
        tp = get_typing_probs(trow)

        p_confused = lp.get("confused", 0.0)
        p_engaged = lp.get("engaged", 0.0)
        p_confident = lp.get("confident", 0.0)

        p_high_load = tp.get("high_load", 0.0) if "high_load" in tp else 0.0

        # Thompson sampling: sample theta from Beta for each persona
        samples = {p: rng.beta(alpha[p], beta[p]) for p in PERSONAS}
        chosen = max(samples.items(), key=lambda kv: kv[1])[0]

        # reward: reduce confusion + reduce load, plus persona effect
        reward = 1.0 - (0.7 * p_confused + 0.3 * p_high_load)
        reward += persona_effect(chosen, p_confused, p_engaged, p_confident, p_high_load)

        # clamp [0, 1]
        reward = float(max(0.0, min(1.0, reward)))

        # update Beta with "soft" Bernoulli: treat reward as success prob
        # sample a binary outcome with probability=reward
        success = 1 if rng.random() < reward else 0
        alpha[chosen] += success
        beta[chosen] += (1 - success)

        running = (running * (ep - 1) + reward) / ep

        # posterior mean for each persona
        probs = {p: float(alpha[p] / (alpha[p] + beta[p])) for p in PERSONAS}

        row = {
            "episode": ep,
            "chosen_persona": chosen,
            "reward": reward,
            "running_avg_reward": running,
        }
        # include persona probability columns (posterior means)
        for p in PERSONAS:
            row[f"prob_{p}"] = probs[p]

        curve_rows.append(row)

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(out_csv, index=False)

    persona_cols = [f"prob_{p}" for p in PERSONAS]
    save_bandit_plot(out_png, curve_df, persona_cols)


# -----------------------------
# Main runner
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Fourthorse Trial Mode - Poster Upgrade Pipeline")
    parser.add_argument("--zip_path", type=str, default="fourthorse_trial_dummy_dataset.zip")
    parser.add_argument("--extract_dir", type=str, default="dataset_unzipped_upgrade")
    parser.add_argument("--outdir", type=str, default="poster_outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    ensure_dir(args.outdir)

    # required subfolders: keep everything inside poster_outputs/
    out_tables = args.outdir
    out_figs = args.outdir
    out_proofs = args.outdir

    # load data
    zip_path = args.zip_path
    if os.path.exists(zip_path):
        paths = unzip_dataset(zip_path, args.extract_dir)
        steps = load_csv(paths.steps_csv)
        choices = load_csv(paths.choices_csv)
        typing = load_csv(paths.typing_csv)
        dataset_hash = sha256_file(zip_path)
    else:
        # fallback to local csvs
        if os.path.exists("data/steps.csv"):
            steps = pd.read_csv("data/steps.csv")
            choices = pd.read_csv("data/choices.csv")
            typing = pd.read_csv("data/typing.csv")
            # hash "dataset" as a hash of the concatenated CSV hashes
            parts = []
            for p in ["data/steps.csv", "data/choices.csv", "data/typing.csv"]:
                parts.append(sha256_file(p))
            dataset_hash = sha256_str("|".join(parts))
        else:
            steps = pd.read_csv("steps.csv")
            choices = pd.read_csv("choices.csv")
            typing = pd.read_csv("typing.csv")
            parts = []
            for p in ["steps.csv", "choices.csv", "typing.csv"]:
                parts.append(sha256_file(p))
            dataset_hash = sha256_str("|".join(parts))

    # validate
    validate_required_columns(steps, ["session_id", "latent_state"], "steps.csv")
    validate_required_columns(choices, ["session_id", "choice_point", "chosen_label"], "choices.csv")
    validate_required_columns(typing, ["session_id", "load_label"], "typing.csv")

    # branch filter
    branch = choices[choices["choice_point"] == "branch"].copy()
    if len(branch) == 0:
        raise ValueError("No branch rows found in choices.csv (choice_point == 'branch').")

    # -------------------------
    # Define feature sets
    # -------------------------
    # Task A (steps -> latent_state)
    steps_label = "latent_state"
    steps_group = "session_id"
    steps_feature_cols = [c for c in steps.columns if c not in (steps_label,)]
    # keep session_id for splitting; drop later from model features automatically
    # also drop obvious non-features if present
    # (but keep screen_id for ablation option)
    # we will pass full columns and let sanitize() and pipeline handle types.

    # Task B (branch choices -> chosen_label)
    choices_label = "chosen_label"
    choices_group = "session_id"
    # only pre-choice features from spec
    choices_feature_cols = [
        "choice_point",  # will drop in run_one_model via feature list - we won't include it here
        "time_to_choice_ms",
        "pre_choice_backtrack_rate",
        "pre_choice_keys_per_sec",
        "pre_choice_mean_pause_ms",
        "pre_choice_terminal_activity",
    ]
    # remove choice_point from features (it’s constant = branch anyway)
    choices_feature_cols = [c for c in choices_feature_cols if c != "choice_point"]

    # Task C (typing -> load_label)
    typing_label = "load_label"
    typing_group = "session_id"
    typing_feature_cols = [c for c in typing.columns if c not in (typing_label,)]

    # -------------------------
    # Run model comparisons (3 tasks x 3 models)
    # -------------------------
    metrics_rows = []

    eval_dir = os.path.join(args.outdir, "_eval_artifacts")
    ensure_dir(eval_dir)

    models = ["majority", "logreg", "rf"]

    # Task A: latent_state (use "both" as main setting)
    for m in models:
        res = run_one_model(
            task="latent_state",
            setting="both",
            df=steps,
            label_col=steps_label,
            feature_cols=[c for c in steps.columns if c not in (steps_label,)],  # include screen_id if present
            group_col=steps_group,
            model_name=m,
            outdir=eval_dir,
            seed=seed,
        )
        metrics_rows.append(res)

    # Task B: chosen_label on branch only
    for m in models:
        res = run_one_model(
            task="chosen_label_branch",
            setting="both",
            df=branch,
            label_col=choices_label,
            feature_cols=choices_feature_cols,
            group_col=choices_group,
            model_name=m,
            outdir=eval_dir,
            seed=seed,
        )
        metrics_rows.append(res)

    # Task C: load_label typing
    for m in models:
        res = run_one_model(
            task="load_label",
            setting="both",
            df=typing,
            label_col=typing_label,
            feature_cols=[c for c in typing.columns if c not in (typing_label,)],
            group_col=typing_group,
            model_name=m,
            outdir=eval_dir,
            seed=seed,
        )
        metrics_rows.append(res)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df_out = os.path.join(out_tables, "metrics_summary.csv")
    metrics_df[["model", "setting", "task", "macro_f1", "acc", "n_train", "n_test"]].to_csv(metrics_df_out, index=False)

    # Figure: model comparison latent_state (macro-F1)
    latent_comp = metrics_df[metrics_df["task"] == "latent_state"].copy()
    save_bar_chart(
        out_png=os.path.join(out_figs, "fig_model_comparison_latent_state.png"),
        title="Latent State — Model Comparison (Macro-F1)",
        labels=latent_comp["model"].tolist(),
        values=latent_comp["macro_f1"].tolist(),
        ylabel="Macro-F1",
    )

    # -------------------------
    # Ablation study (latent_state only)
    # -------------------------
    step_cols_all = [c for c in steps.columns if c not in (steps_label,)]
    # infer nav vs typing groups (excluding identifiers)
    step_cols_no_ids = [c for c in step_cols_all if c not in ("session_id",)]
    nav_cols, typ_cols = infer_feature_groups(step_cols_no_ids)

    ablation_settings = [
        ("nav_only", nav_cols),
        ("typing_only", typ_cols),
        ("both", step_cols_no_ids),
        ("both_without_screen_id", [c for c in step_cols_no_ids if c != "screen_id"]),
    ]

    ab_rows = []
    for setting_name, feat_cols in ablation_settings:
        # ensure we keep session_id for splitting
        feat_cols_for_run = feat_cols.copy()
        res = run_one_model(
            task="latent_state",
            setting=setting_name,
            df=steps,
            label_col=steps_label,
            feature_cols=feat_cols_for_run + ["session_id"],  # safe; sanitize will keep only if exists
            group_col=steps_group,
            model_name="logreg",  # ablation uses one consistent model
            outdir=eval_dir,
            seed=seed,
        )
        ab_rows.append({
            "task": "latent_state",
            "setting": setting_name,
            "macro_f1": res["macro_f1"],
        })

    ab_df = pd.DataFrame(ab_rows)
    ab_df_out = os.path.join(out_tables, "ablation_results.csv")
    ab_df.to_csv(ab_df_out, index=False)

    save_bar_chart(
        out_png=os.path.join(out_figs, "fig_ablation_latent_state.png"),
        title="Latent State — Ablation (Macro-F1)",
        labels=ab_df["setting"].tolist(),
        values=ab_df["macro_f1"].tolist(),
        ylabel="Macro-F1",
    )

    # -------------------------
    # Train final models for trajectory + bandit (use RF for latent_state + RF for typing)
    # -------------------------
    # Final latent_state model (RF)
    latent_final = run_one_model(
        task="latent_state",
        setting="both",
        df=steps,
        label_col=steps_label,
        feature_cols=[c for c in steps.columns if c not in (steps_label,)],
        group_col=steps_group,
        model_name="rf",
        outdir=eval_dir,
        seed=seed,
    )

    # Reload trained model (we saved only logreg/rf latent_state both, but we saved model file only for logreg/rf latent_state both)
    # Here: save explicitly
    latent_model_path = os.path.join(eval_dir, "latent_state__both__rf__model.joblib")
    # train + save
    df2 = sanitize(steps, allowed_cols=["session_id", steps_label] + [c for c in steps.columns if c != steps_label]).dropna(subset=[steps_label])
    tr_idx, te_idx = group_split_idx(df2, label_col=steps_label, group_col="session_id", seed=seed)
    tr = df2.iloc[tr_idx].copy()
    te = df2.iloc[te_idx].copy()
    y_tr = tr[steps_label].astype(str)
    X_tr = tr.drop(columns=[steps_label])
    y_te = te[steps_label].astype(str)
    X_te = te.drop(columns=[steps_label])

    if "session_id" in X_tr.columns:
        X_tr_m = X_tr.drop(columns=["session_id"])
    else:
        X_tr_m = X_tr
    if "session_id" in X_te.columns:
        X_te_m = X_te.drop(columns=["session_id"])
    else:
        X_te_m = X_te

    latent_pipe = build_pipeline(X_tr_m, model_name="rf", seed=seed)
    latent_pipe.fit(X_tr_m, y_tr)
    joblib.dump(latent_pipe, latent_model_path)

    # Final typing model (RF) for bandit
    typing_df2 = sanitize(typing, allowed_cols=["session_id", typing_label] + [c for c in typing.columns if c != typing_label]).dropna(subset=[typing_label])
    tr_idx_t, te_idx_t = group_split_idx(typing_df2, label_col=typing_label, group_col="session_id", seed=seed)
    trt = typing_df2.iloc[tr_idx_t].copy()
    tet = typing_df2.iloc[te_idx_t].copy()
    y_trt = trt[typing_label].astype(str)
    X_trt = trt.drop(columns=[typing_label])
    if "session_id" in X_trt.columns:
        X_trt_m = X_trt.drop(columns=["session_id"])
    else:
        X_trt_m = X_trt
    typing_pipe = build_pipeline(X_trt_m, model_name="rf", seed=seed)
    typing_pipe.fit(X_trt_m, y_trt)
    typing_model_path = os.path.join(eval_dir, "load_label__both__rf__model.joblib")
    joblib.dump(typing_pipe, typing_model_path)

    # -------------------------
    # Trajectory plot for one example session (from test split)
    # -------------------------
    # pick a session from test split
    test_sessions = te["session_id"].unique().tolist() if "session_id" in te.columns else []
    if len(test_sessions) == 0:
        # fallback: pick any session
        test_sessions = df2["session_id"].unique().tolist()

    example_sid = test_sessions[0]
    ex = df2[df2["session_id"] == example_sid].copy()
    ex = ex.sort_values("step_idx") if "step_idx" in ex.columns else ex

    # proba requires model features (no session_id, no label)
    exX = ex.drop(columns=[steps_label], errors="ignore")
    if "session_id" in exX.columns:
        exX = exX.drop(columns=["session_id"])

    # keep only columns used in training
    exX = exX.reindex(columns=X_tr_m.columns, fill_value=np.nan)

    proba = latent_pipe.predict_proba(exX)
    class_names = list(latent_pipe.named_steps["clf"].classes_)

    save_trajectory_plot(
        out_png=os.path.join(out_figs, "fig_state_trajectory_example.png"),
        session_df=ex,
        proba=proba,
        class_names=class_names,
    )

    # -------------------------
    # Bandit closed-loop self-upgrade demo
    # -------------------------
    bandit_csv = os.path.join(out_tables, "bandit_learning_curve.csv")
    bandit_png = os.path.join(out_figs, "fig_bandit_convergence.png")

    # For bandit, define feature cols exactly as used in pipelines
    latent_feature_cols = list(X_tr.columns)  # includes session_id sometimes; bandit functions drop session_id
    typing_feature_cols = list(X_trt.columns)

    thompson_bandit_sim(
        steps_df=df2,
        latent_pipe=latent_pipe,
        latent_feature_cols=latent_feature_cols,
        typing_df=typing_df2,
        typing_pipe=typing_pipe,
        typing_feature_cols=typing_feature_cols,
        out_csv=bandit_csv,
        out_png=bandit_png,
        seed=seed,
        episodes=300,
    )

    # -------------------------
    # Commitments + privacy spec (black-box proof bundle)
    # -------------------------
    feature_schema = sorted([c for c in X_tr_m.columns])
    feature_schema_hash = sha256_str("|".join(feature_schema))

    policy_hash = sha256_str(POLICY_CONFIG_STR)

    model_hashes = {
        "latent_state_model": sha256_file(latent_model_path),
        "typing_load_model": sha256_file(typing_model_path),
    }

    commitments = {
        "dataset_hash": dataset_hash,
        "feature_schema_hash": feature_schema_hash,
        "policy_hash": policy_hash,
        "model_hashes": model_hashes,
    }
    write_json(os.path.join(out_proofs, "commitments.json"), commitments)

    privacy_spec = "\n".join([
        "- No raw text collected.",
        "- No raw keystroke event streams stored.",
        "- Only aggregated features per step/episode are used (dwell, backtrack, typing rates, pause stats, etc.).",
        "- session_id is used only for group train/test splitting and removed from model features.",
        "- Predictions/probabilities and persona selections can be logged without revealing raw content.",
        "- Models and policies are versioned by SHA-256 hashes for auditability (black-box proof bundle).",
        "",
        "Allowed logs: aggregated telemetry features, predicted probabilities, chosen persona, model version hash.",
        "Never logged: raw terminal text, raw keystroke events, or any event-level payloads.",
    ])
    write_text(os.path.join(out_proofs, "privacy_spec.txt"), privacy_spec)

    # Summary print
    print("\n=== poster_outputs generated ===")
    print(f"Output dir: {os.path.abspath(args.outdir)}")
    print(f"metrics_summary.csv: {metrics_df_out}")
    print(f"ablation_results.csv: {ab_df_out}")
    print(f"bandit_learning_curve.csv: {bandit_csv}")
    print("Figures: fig_ablation_latent_state.png, fig_model_comparison_latent_state.png, fig_state_trajectory_example.png, fig_bandit_convergence.png")
    print("Proofs: commitments.json, privacy_spec.txt")


if __name__ == "__main__":
    main()
EOF
