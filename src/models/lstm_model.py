import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):

    def __init__(
        self,
        input_size=11,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ====================================================
        # LSTM BACKBONE
        # ====================================================

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # ====================================================
        # FORECASTING HEAD
        # ====================================================

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        # x shape:
        # (batch, sequence_length, features)

        lstm_output, (hidden, cell) = self.lstm(x)

        # Take the final time-step representation
        last_hidden = lstm_output[:, -1, :]

        # Forecast
        output = self.fc(last_hidden)

        # Remove final dimension
        output = output.squeeze(1)

        return output