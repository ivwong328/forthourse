import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from models.features import STEPS_FEATURES, STEPS_TARGET
from models.metrics import print_clf_metrics

LABELS = ["confused","engaged","confident"]

def main():
    df = pd.read_csv("data/steps.csv")

    # Basic cleaning: drop rows missing target/features
    df = df.dropna(subset=STEPS_FEATURES + [STEPS_TARGET, "session_id"])

    X = df[STEPS_FEATURES]
    y = df[STEPS_TARGET].astype(str)
    groups = df["session_id"].astype(str)

    # Split by session to avoid leakage across steps within same session
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Baseline 1: Logistic Regression (scaled)
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto"))
    ])
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print_clf_metrics(y_test, y_pred, labels=LABELS, title="Dataset A - LogisticRegression")

    # Baseline 2: Random Forest (often strong with mixed features)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print_clf_metrics(y_test, y_pred_rf, labels=LABELS, title="Dataset A - RandomForest")

    # Feature importance (RF)
    importances = rf.feature_importances_
    top = sorted(zip(STEPS_FEATURES, importances), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 RF feature importances:")
    for f, v in top:
        print(f"  {f:24s} {v:.4f}")

if __name__ == "__main__":
    main()

