import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from models.features import STEPS_FEATURES

def fit_hmm(steps_df: pd.DataFrame, n_states=3, random_state=42):
    steps_df = steps_df.sort_values(["session_id","step_idx"])
    X = steps_df[STEPS_FEATURES].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # lengths for sequences per session
    lengths = steps_df.groupby("session_id").size().tolist()

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        random_state=random_state
    )
    hmm.fit(Xs, lengths=lengths)

    states = hmm.predict(Xs, lengths=lengths)
    return hmm, scaler, states
