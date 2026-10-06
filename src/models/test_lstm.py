import torch

from lstm_model import LSTMForecaster


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
# CREATE MODEL
# ============================================================

model = LSTMForecaster(
    input_size=11,
    hidden_size=128,
    num_layers=2,
    dropout=0.2
)

model = model.to(device)


print("\nModel:")
print(model)


# ============================================================
# CREATE SAMPLE INPUT
# ============================================================

batch_size = 32
sequence_length = 30
features = 11

x = torch.randn(
    batch_size,
    sequence_length,
    features
).to(device)


print("\nInput shape:")
print(x.shape)

print(
    "Input device:",
    x.device
)


# ============================================================
# FORWARD PASS
# ============================================================

with torch.no_grad():

    predictions = model(x)


print("\nPrediction shape:")
print(predictions.shape)

print(
    "Prediction device:",
    predictions.device
)


# ============================================================
# CHECK GPU MEMORY
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


print("\nLSTM model test successful! ✅")