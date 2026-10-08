# Enhanced NMT-Net for Multi-Task Demand Forecasting

An enhanced neuroplasticity-inspired multi-task demand forecasting framework based on the IEEE paper:

**"Neuroplasticity-inspired dynamic ANNs for multi-task demand forecasting"**

## Project Objective

This project enhances the base NMT-Net framework with:

1. LSTM-based temporal learning
2. Learned task embeddings
3. Explicit catastrophic forgetting evaluation

## Dataset

The project uses the **Store Item Demand Forecasting** dataset.

- Records: 913,000
- Stores: 10
- Items: 50
- Store × Item Tasks: 500
- Historical Window: 30 days
- Target: Next-day demand forecasting

Each Store × Item combination is treated as one forecasting task.

The raw dataset is not included in this repository.

## Project Workflow

```text
Demand Data
     ↓
Data Preprocessing
     ↓
Task Creation
     ↓
30-Day Sequence Generation
     ↓
LSTM Temporal Learning
     ↓
Task Embedding
     ↓
Next-Day Demand Prediction
     ↓
Model Evaluation
     ↓
Sequential Learning
     ↓
Catastrophic Forgetting Analysis
````

## Data Pipeline

The data preprocessing pipeline includes:

* Data loading
* Date conversion
* Duplicate checking
* Missing-value checking
* Chronological sorting
* Temporal feature engineering
* Store × Item task creation
* Chronological train/validation/test split
* Train-only feature scaling
* 30-day sliding-window sequence generation

### Dataset Split

* Training: 639,000 rows
* Validation: 137,000 rows
* Testing: 137,000 rows

## Sequence Data

The model uses the previous 30 days to predict the next day's demand.

```text
X shape      : (624000, 30, 11)
y shape      : (624000,)
Task IDs     : (624000,)
```

The 11 input features include:

* Sales
* Month
* Day
* Day of Week
* Week of Year
* Day of Year
* Weekend
* Month Sin
* Month Cos
* Day-of-Week Sin
* Day-of-Week Cos

## LSTM Model

A 2-layer LSTM is used to learn temporal patterns from historical demand.

### Configuration

* Input Features: 11
* Hidden Size: 128
* LSTM Layers: 2
* Dropout: 0.2

The LSTM processes the 30-day historical sequence and predicts the next day's demand.

## Task Embedding

Each Store × Item combination is assigned a unique Task ID.

* Number of Tasks: 500
* Embedding Dimension: 32

The task embedding provides task-specific information to the forecasting model.

```text
LSTM Representation
        +
Task Embedding
        ↓
Combined Representation
        ↓
Prediction Head
        ↓
Next-Day Demand
```

## LSTM Baseline Results

| Metric |   Result |
| ------ | -------: |
| MAE    |   6.6386 |
| RMSE   |   8.6604 |
| MAPE   | 12.5821% |
| sMAPE  | 12.0671% |
| R²     |   0.9294 |

## LSTM + Task Embedding Results

| Metric |   Result |
| ------ | -------: |
| MAE    |   6.6573 |
| RMSE   |   8.6966 |
| MAPE   | 12.3019% |
| sMAPE  | 12.1236% |
| R²     |   0.9288 |

### Model Comparison

| Model                 |    MAE |   RMSE |     MAPE |    sMAPE |     R² |
| --------------------- | -----: | -----: | -------: | -------: | -----: |
| LSTM Baseline         | 6.6386 | 8.6604 | 12.5821% | 12.0671% | 0.9294 |
| LSTM + Task Embedding | 6.6573 | 8.6966 | 12.3019% | 12.1236% | 0.9288 |

The task embedding experiment improved MAPE, while the other metrics were slightly different from the LSTM baseline.

## Catastrophic Forgetting Evaluation

The 500 tasks are divided into 10 sequential groups of 50 tasks.

```text
Stage 1  → Tasks 1–50
Stage 2  → Tasks 1–100
Stage 3  → Tasks 1–150
...
Stage 10 → Tasks 1–500
```

After learning new tasks, previously learned tasks are evaluated again.

### Forgetting Formula

```text
Forgetting = Current RMSE - Best Previous RMSE
```

### Final Results

* Final Stage RMSE: 8.5978
* Mean Forgetting: 0.1485
* Maximum Forgetting: 2.1256
* Tasks with Forgetting: 362 / 500

This experiment shows how sequential learning can affect previously learned tasks.

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* CUDA
* Git
* GitHub

## GPU Environment

```text
PyTorch : 2.11.0+cu128
CUDA    : 12.8
GPU     : NVIDIA GeForce RTX 3050 Laptop GPU
```

## Project Structure

```text
NMT-Net-Enhanced/
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       ├── train_processed.csv
│       ├── val_processed.csv
│       ├── test_processed.csv
│       ├── task_mapping.csv
│       └── scaler.pkl
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── sequence_builder.py
│   │   ├── dataset.py
│   │   └── test_dataloader.py
│   │
│   ├── models/
│   │   ├── lstm_model.py
│   │   ├── task_embedding.py
│   │   ├── task_embedding_model.py
│   │   └── ...
│   │
│   ├── train_lstm.py
│   ├── train_task_embedding.py
│   └── catastrophic_forgetting.py
│
├── models/
│   ├── lstm_baseline_best.pt
│   ├── task_embedding_best.pt
│   └── forgetting_final.pt
│
├── results/
│   ├── lstm_baseline_metrics.csv
│   ├── lstm_baseline_predictions.csv
│   ├── task_embedding_metrics.csv
│   ├── task_embedding_predictions.csv
│   ├── forgetting_stage_metrics.csv
│   ├── forgetting_task_summary.csv
│   └── forgetting_curve.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_results_final.ipynb
│
└── README.md
```

## Project Status

| Component                  | Status    |
| -------------------------- | --------- |
| Data preprocessing         | Completed |
| Feature engineering        | Completed |
| Task creation              | Completed |
| Data splitting             | Completed |
| Data scaling               | Completed |
| Sequence generation        | Completed |
| PyTorch Dataset/DataLoader | Completed |
| CUDA configuration         | Completed |
| LSTM implementation        | Completed |
| LSTM training              | Completed |
| Task embeddings            | Completed |
| Task embedding training    | Completed |
| Forgetting evaluation      | Completed |
| Results generation         | Completed |
| Model comparison           | Completed |

## Conclusion

The project enhances the base NMT-Net framework using LSTM-based temporal learning, learned task embeddings, and explicit catastrophic forgetting evaluation.

The LSTM model achieved strong forecasting performance, while the forgetting experiment showed that sequential learning can affect previously learned tasks.

## Future Work

* Improve the dynamic task adaptation mechanism.
* Develop methods to reduce catastrophic forgetting.
* Explore Attention and Transformer-based models.
* Test the framework on larger datasets.
* Extend the model toward real-time demand forecasting and deployment.

## Reference

**Neuroplasticity-inspired dynamic ANNs for multi-task demand forecasting**

Base Paper Repository:

[https://github.com/MatZar01/Multi_Forecasting](https://github.com/MatZar01/Multi_Forecasting)

```
```
