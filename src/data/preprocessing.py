import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):
    print("\n[1] Loading dataset...")

    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))

    return df


# ============================================================
# 2. BASIC CLEANING
# ============================================================

def basic_cleaning(df):
    print("\n[2] Cleaning data...")

    df = df.copy()

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print("Duplicates removed:", before - after)

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Remove rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Sort chronologically for every store-item pair
    df = df.sort_values(
        ["store", "item", "date"]
    ).reset_index(drop=True)

    return df


# ============================================================
# 3. CREATE TEMPORAL FEATURES
# ============================================================

def create_time_features(df):
    print("\n[3] Creating temporal features...")

    df = df.copy()

    # Basic calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclic month encoding
    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # Cyclic day-of-week encoding
    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    print("Temporal features created.")

    return df


# ============================================================
# 4. CREATE TASK IDs
# ============================================================

def create_task_id(df):
    print("\n[4] Creating task IDs...")

    df = df.copy()

    # Each store-item combination = one task
    df["task_id"] = (
        df["store"].astype(str)
        + "_"
        + df["item"].astype(str)
    )

    # Convert task names to integer IDs
    unique_tasks = sorted(
        df["task_id"].unique()
    )

    task_to_idx = {
        task: idx
        for idx, task in enumerate(unique_tasks)
    }

    df["task_idx"] = df["task_id"].map(
        task_to_idx
    )

    # Mapping table
    task_mapping = pd.DataFrame(
        {
            "task_id": unique_tasks,
            "task_idx": range(len(unique_tasks))
        }
    )

    print("Number of tasks:", len(unique_tasks))

    return df, task_mapping


# ============================================================
# 5. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
# ============================================================

def chronological_split(
    df,
    train_ratio=0.70,
    val_ratio=0.15
):

    print("\n[5] Performing chronological split...")

    train_parts = []
    val_parts = []
    test_parts = []

    # Split independently for every task
    for task_id, task_df in df.groupby("task_id"):

        task_df = task_df.sort_values("date")

        n = len(task_df)

        train_end = int(
            n * train_ratio
        )

        val_end = int(
            n * (train_ratio + val_ratio)
        )

        train_parts.append(
            task_df.iloc[:train_end]
        )

        val_parts.append(
            task_df.iloc[train_end:val_end]
        )

        test_parts.append(
            task_df.iloc[val_end:]
        )

    train_df = pd.concat(
        train_parts
    ).reset_index(drop=True)

    val_df = pd.concat(
        val_parts
    ).reset_index(drop=True)

    test_df = pd.concat(
        test_parts
    ).reset_index(drop=True)

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    return train_df, val_df, test_df


# ============================================================
# 6. SCALE NUMERICAL FEATURES
# ============================================================

def scale_data(
    train_df,
    val_df,
    test_df
):

    print("\n[6] Scaling numerical features...")

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Features used by the model
    scale_columns = [
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

    # Fit ONLY on training data
    scaler = StandardScaler()

    train_df[scale_columns] = scaler.fit_transform(
        train_df[scale_columns]
    )

    # Transform validation and test
    val_df[scale_columns] = scaler.transform(
        val_df[scale_columns]
    )

    test_df[scale_columns] = scaler.transform(
        test_df[scale_columns]
    )

    print("Scaler fitted on TRAIN data only.")

    return train_df, val_df, test_df, scaler


# ============================================================
# 7. COMPLETE PIPELINE
# ============================================================

def preprocess_pipeline(
    input_path,
    output_dir="data/processed"
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Load
    df = load_data(input_path)

    # Clean
    df = basic_cleaning(df)

    # Temporal features
    df = create_time_features(df)

    # Task IDs
    df, task_mapping = create_task_id(df)

    # Chronological split
    train_df, val_df, test_df = chronological_split(
        df
    )

    # Scaling
    train_df, val_df, test_df, scaler = scale_data(
        train_df,
        val_df,
        test_df
    )

    # Save processed datasets
    train_path = os.path.join(
        output_dir,
        "train_processed.csv"
    )

    val_path = os.path.join(
        output_dir,
        "val_processed.csv"
    )

    test_path = os.path.join(
        output_dir,
        "test_processed.csv"
    )

    mapping_path = os.path.join(
        output_dir,
        "task_mapping.csv"
    )

    train_df.to_csv(
        train_path,
        index=False
    )

    val_df.to_csv(
        val_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    task_mapping.to_csv(
        mapping_path,
        index=False
    )

    print("\nProcessed files saved:")
    print(train_path)
    print(val_path)
    print(test_path)
    print(mapping_path)

    return (
        train_df,
        val_df,
        test_df,
        task_mapping,
        scaler
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_path = "data/raw/train.csv"

    preprocess_pipeline(
        input_path
    )