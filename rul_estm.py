import pandas as pd

column_names = [
    "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

# dokimi gia to arxeio train_FD001.txt
train_path = "CMAPSSData/train_FD001.txt"
train_df = pd.read_csv(train_path, sep="\s+", header=None, names=column_names)

# ypologismos tou rul estm me max cycle - cycle 
rul_df = train_df.groupby("unit")["cycle"].max().reset_index()
rul_df.columns = ["unit", "max_cycle"]
train_df = train_df.merge(rul_df, on="unit")
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
train_df.drop("max_cycle", axis=1, inplace=True)

print(train_df.head(50))
