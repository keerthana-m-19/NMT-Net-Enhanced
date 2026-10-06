import pandas as pd
import torch

from torch.utils.data import DataLoader

from sequence_builder import build_sequences
from dataset import DemandForecastDataset


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD PROCESSED TRAINING DATA
# ============================================================

print("\nLoading training data...")

train_df = pd.read_csv(
    "data/processed/train_processed.csv",
    parse_dates=["date"]
)


# ============================================================
# BUILD SEQUENCES
# ============================================================

X, y, task_ids, dates = build_sequences(
    train_df
)


# ============================================================
# CREATE DATASET
# ============================================================

print("\nCreating PyTorch Dataset...")

dataset = DemandForecastDataset(
    X,
    y,
    task_ids
)

print(
    "Dataset size:",
    len(dataset)
)


# ============================================================
# CREATE DATALOADER
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


print(
    "Number of batches:",
    len(loader)
)


# ============================================================
# TEST ONE BATCH
# ============================================================

print("\nTesting first batch...")

batch = next(iter(loader))

sequence = batch["sequence"]
target = batch["target"]
task_id = batch["task_id"]


print(
    "Sequence shape:",
    sequence.shape
)

print(
    "Target shape:",
    target.shape
)

print(
    "Task ID shape:",
    task_id.shape
)


# ============================================================
# MOVE BATCH TO GPU
# ============================================================

sequence = sequence.to(
    device,
    non_blocking=True
)

target = target.to(
    device,
    non_blocking=True
)

task_id = task_id.to(
    device,
    non_blocking=True
)


print("\nAfter moving to GPU:")

print(
    "Sequence device:",
    sequence.device
)

print(
    "Target device:",
    target.device
)

print(
    "Task ID device:",
    task_id.device
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**2
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**2
    )

    print(
        f"\nGPU memory allocated: "
        f"{allocated:.2f} MB"
    )

    print(
        f"GPU memory reserved: "
        f"{reserved:.2f} MB"
    )


print("\nDataLoader test successful! ✅")