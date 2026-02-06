import pandas as pd

steps_df   = pd.read_csv("steps.csv")
choices_df = pd.read_csv("choices.csv")
typing_df  = pd.read_csv("typing.csv")

print("STEPS:")
print(steps_df.head(), "\n")

print("CHOICES:")
print(choices_df.head(), "\n")

print("TYPING:")
print(typing_df.head(), "\n")
