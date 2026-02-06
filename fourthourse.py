import pandas as pd

steps_df   = pd.read_csv("steps.csv")
choices_df = pd.read_csv("choices.csv")
typing_df  = pd.read_csv("typing.csv")

print(steps_df.head())
print(choices_df.head())
print(typing_df.head())


# Ensure fixed column order
steps_order = [
    "session_id","step_idx","screen_id","time_since_start_ms","dwell_ms",
    "nav_next_count","nav_back_count","backtrack_rate",
    "terminal_submit_count","terminal_chars_total","unique_tokens","archive_add_count",
    "keys_total","keys_per_sec","backspace_count","backspace_rate","enter_count",
    "pause_count","mean_pause_ms","burstiness","hesitation_score","confusion_proxy","latent_state"
]

choices_order = [
    "session_id","choice_point","time_to_choice_ms","pre_choice_backtrack_rate",
    "pre_choice_keys_per_sec","pre_choice_mean_pause_ms","pre_choice_terminal_activity",
    "chosen_label","chosen_idx"
]

typing_order = [
    "session_id","episode_idx","screen_id","text_len","keys_total","keys_per_sec",
    "backspace_count","backspace_rate","pause_count","mean_pause_ms","rewrite_ratio",
    "certainty_proxy","load_label"
]

# steps_df = steps_df[steps_order]
# choices_df = choices_df[choices_order]
# typing_df = typing_df[typing_order]

steps_df.to_csv("steps.csv", index=False)
choices_df.to_csv("choices.csv", index=False)
typing_df.to_csv("typing.csv", index=False)

# Optional: zip them
# import zipfile
# with zipfile.ZipFile("fourthorse_dummy_export.zip", "w") as z:
#     z.write("steps.csv")
#     z.write("choices.csv")
#     z.write("typing.csv")
