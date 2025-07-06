import pandas as pd

def load_train_data(path):
    column_names = [
        "unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]

    train_df = pd.read_csv(f"{path}/train_FD002.txt", sep=r"\s+", header=None, names=column_names)

    rul_df = train_df.groupby("unit")["cycle"].max().reset_index()
    rul_df.columns = ["unit", "max_cycle"]
    train_df = train_df.merge(rul_df, on="unit")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop("max_cycle", axis=1, inplace=True)

    return train_df, column_names

def load_test_data(path, column_names):
    test_df = pd.read_csv(f"{path}/test_FD002.txt", sep=r"\s+", header=None, names=column_names)
    rul_true = pd.read_csv(f"{path}/RUL_FD002.txt", header=None).values.flatten()
    return test_df, rul_true