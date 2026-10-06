# Enhanced NMT-Net for Multi-Task Demand Forecasting

An enhanced neuroplasticity-inspired multi-task demand forecasting framework based on the IEEE paper:

**"Neuroplasticity-inspired dynamic ANNs for multi-task demand forecasting"**

## Project Objective

This project enhances the base NMT-Net approach with:

1. LSTM-based temporal learning
2. Learned task embeddings
3. Explicit catastrophic forgetting evaluation

## Dataset

The project uses the Store Item Demand Forecasting dataset.

- Records: 913,000
- Stores: 10
- Items: 50
- Store × Item tasks: 500
- Historical window: 30 days
- Target: Next-day demand forecasting

The raw dataset is not included in this repository.

## Completed Work

### Data Pipeline

- Data loading
- Date conversion
- Duplicate checking
- Missing-value checking
- Chronological sorting
- Temporal feature engineering
- Store × Item task creation
- Chronological train/validation/test split
- Train-only feature scaling
- 30-day sliding-window sequence generation

### Sequence Data

Training sequences:

- X shape: `(624000, 30, 11)`
- y shape: `(624000,)`
- Task IDs: `(624000,)`

### PyTorch Pipeline

- PyTorch Dataset implemented
- DataLoader implemented
- CUDA support verified
- NVIDIA RTX 3050 Laptop GPU tested

### LSTM

Implemented:

- 2-layer LSTM
- Hidden size: 128
- Dropout: 0.2
- Fully connected prediction head

The LSTM forward pass has been successfully tested on the NVIDIA GPU.

## Current Status

| Component | Status |
|---|---|
| Data preprocessing | Completed |
| Feature engineering | Completed |
| Task creation | Completed |
| Data splitting | Completed |
| Data scaling | Completed |
| Sequence generation | Completed |
| PyTorch Dataset/DataLoader | Completed |
| CUDA configuration | Completed |
| LSTM implementation | Completed |
| LSTM GPU forward test | Completed |
| LSTM training | Next |
| Task embeddings | Planned |
| Dynamic adaptation | Planned |
| Continual learning | Planned |
| Forgetting evaluation | Planned |
| Final comparison | Planned |

## Project Architecture

```text
Store × Item Demand Data
          ↓
Data Preprocessing
          ↓
Temporal Feature Engineering
          ↓
Task Creation
          ↓
Chronological Split
          ↓
Train-only Scaling
          ↓
30-Day Sliding Window
          ↓
LSTM Backbone
          ↓
Learned Task Embedding
          ↓
Task-Aware Fusion
          ↓
Dynamic Adaptation
          ↓
Demand Forecast
          ↓
Continual Learning
          ↓
Catastrophic Forgetting Evaluation