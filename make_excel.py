import pandas as pd

steps_df   = pd.read_csv("steps.csv")
choices_df = pd.read_csv("choices.csv")
typing_df  = pd.read_csv("typing.csv")

with pd.ExcelWriter("fourthorse_dataset.xlsx", engine="openpyxl") as writer:
    steps_df.to_excel(writer, sheet_name="Steps", index=False)
    choices_df.to_excel(writer, sheet_name="Choices", index=False)
    typing_df.to_excel(writer, sheet_name="Typing", index=False)

print("Saved: fourthorse_dataset.xlsx")
