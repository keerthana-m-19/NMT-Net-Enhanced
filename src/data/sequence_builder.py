import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

LAG = 30

FEATURE_COLUMNS = [
    "sales",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "is_weekend",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos"
]


# ============================================================
# CREATE SEQUENCES FOR ONE TASK
# ============================================================

def create_sequences_for_task(
    task_df,
    feature_columns=FEATURE_COLUMNS,
    target_column="sales",
    lag=LAG
):

    task_df = task_df.sort_values("date").reset_index(drop=True)

    features = task_df[feature_columns].values.astype(
        np.float32
    )

    targets = task_df[target_column].values.astype(
        np.float32
    )

    task_idx = task_df["task_idx"].iloc[0]

    X = []
    y = []
    task_ids = []
    target_dates = []

    for i in range(lag, len(task_df)):

        # Previous 30 days
        sequence = features[
            i - lag:i
        ]

        # Next day's demand
        target = targets[i]

        X.append(sequence)
        y.append(target)
        task_ids.append(task_idx)

        target_dates.append(
            task_df["date"].iloc[i]
        )

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(task_ids, dtype=np.int64),
        np.array(target_dates)
    )


# ============================================================
# BUILD SEQUENCES FOR ALL TASKS
# ============================================================

def build_sequences(
    df,
    feature_columns=FEATURE_COLUMNS,
    target_column="sales",
    lag=LAG
):

    print("\nBuilding sequences...")

    X_all = []
    y_all = []
    task_ids_all = []
    dates_all = []

    for task_id, task_df in df.groupby(
        "task_id"
    ):

        X, y, task_ids, dates = create_sequences_for_task(
            task_df,
            feature_columns,
            target_column,
            lag
        )

        X_all.append(X)
        y_all.append(y)
        task_ids_all.append(task_ids)
        dates_all.append(dates)

    X_all = np.concatenate(X_all)
    y_all = np.concatenate(y_all)
    task_ids_all = np.concatenate(task_ids_all)
    dates_all = np.concatenate(dates_all)

    print("\nSequence creation completed.")

    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)
    print("Task IDs shape:", task_ids_all.shape)
    print("Dates shape:", dates_all.shape)

    return (
        X_all,
        y_all,
        task_ids_all,
        dates_all
    )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("Loading processed training data...")

    train_path = (
        "data/processed/train_processed.csv"
    )

    train_df = pd.read_csv(
        train_path,
        parse_dates=["date"]
    )

    X, y, task_ids, dates = build_sequences(
        train_df
    )

    print("\nFinal verification:")
    print("--------------------")

    print("X:", X.shape)
    print("y:", y.shape)
    print("task_ids:", task_ids.shape)

    print("\nExample sequence:")
    print(X[0])

    print("\nExample target:")
    print(y[0])

    print("\nExample task ID:")
    print(task_ids[0])

    print("\nExample target date:")
    print(dates[0])