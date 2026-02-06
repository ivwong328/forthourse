from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def print_clf_metrics(y_true, y_pred, labels=None, title=""):
    if title:
        print(f"\n=== {title} ===")
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))

from __future__ import annotations

# Dataset A (steps) features: DO NOT include latent_state
STEPS_FEATURES = [
    "dwell_ms","nav_next_count","nav_back_count","backtrack_rate",
    "terminal_submit_count","terminal_chars_total","unique_tokens","archive_add_count",
    "keys_total","keys_per_sec","backspace_count","backspace_rate","enter_count",
    "pause_count","mean_pause_ms","burstiness","hesitation_score","confusion_proxy"
]
STEPS_TARGET = "latent_state"

# Dataset B (choices) features: DO NOT include chosen_label/chosen_idx
CHOICES_FEATURES = [
    "time_to_choice_ms","pre_choice_backtrack_rate","pre_choice_keys_per_sec",
    "pre_choice_mean_pause_ms","pre_choice_terminal_activity"
]
CHOICES_TARGET = "chosen_idx"

# Dataset C (typing) features: DO NOT include load_label
TYPING_FEATURES = [
    "text_len","keys_total","keys_per_sec","backspace_count","backspace_rate",
    "pause_count","mean_pause_ms","rewrite_ratio","certainty_proxy"
]
TYPING_TARGET = "load_label"
