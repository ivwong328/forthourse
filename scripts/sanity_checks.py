import pandas as pd

def check_steps(df: pd.DataFrame):
    print("\n[STEPS] rows:", len(df), "sessions:", df["session_id"].nunique())
    # time monotonic within session
    bad = 0
    for sid, g in df.sort_values(["session_id","step_idx"]).groupby("session_id"):
        if not (g["time_since_start_ms"].is_monotonic_increasing):
            bad += 1
    print("[STEPS] sessions with non-monotonic time_since_start_ms:", bad)

    # screen-24 presence per session
    hub_counts = df.groupby("session_id")["screen_id"].apply(lambda s: (s=="screen-24").sum())
    print("[STEPS] screen-24 count stats:", hub_counts.describe())
    print("[STEPS] sessions missing screen-24:", (hub_counts==0).sum())

def check_choices(df: pd.DataFrame):
    print("\n[CHOICES] rows:", len(df), "sessions:", df["session_id"].nunique())
    per = df.groupby("session_id")["choice_point"].nunique()
    print("[CHOICES] sessions with !=2 unique choice_points:", (per!=2).sum())
    print("[CHOICES] choice_point counts:\n", df["choice_point"].value_counts())

def check_typing(df: pd.DataFrame):
    print("\n[TYPING] rows:", len(df), "sessions:", df["session_id"].nunique())
    per = df.groupby("session_id")["episode_idx"].count()
    print("[TYPING] episodes per session stats:", per.describe())

if __name__ == "__main__":
    steps = pd.read_csv("data/steps.csv")
    choices = pd.read_csv("data/choices.csv")
    typing = pd.read_csv("data/typing.csv")

    check_steps(steps)
    check_choices(choices)
    check_typing(typing)
