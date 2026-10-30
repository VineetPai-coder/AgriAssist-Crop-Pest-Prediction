import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron Baseline.
    Operates on flattened window vectors of shape [batch_size, 14 * num_features].
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout_rate=0.3):
        super(MLPModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, input_dim]
        return self.network(x)


class CNN1DModel(nn.Module):
    """
    1D Convolutional Neural Network Variant.
    Extracts cross-channel localized temporal features from input [batch_size, seq_len, num_features].
    """
    def __init__(self, num_features, seq_len=14, out_channels1=32, out_channels2=64, dropout_rate=0.3):
        super(CNN1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=out_channels1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels2)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(out_channels2, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, num_features]
        # Transpose to [batch_size, num_features, seq_len] for Conv1D
        x = x.transpose(1, 2)
        
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x) # Shape: [batch_size, out_channels2, 1]
        x = torch.flatten(x, 1) # Shape: [batch_size, out_channels2]
        return self.fc(x)


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) Sequence Model.
    Captures long-range microclimate temporal dependencies from sequence [batch_size, seq_len, num_features].
    """
    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use final time-step hidden state representation
        last_hidden = lstm_out[:, -1, :] # Shape: [batch_size, hidden_dim]
        return self.fc(last_hidden)


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) Sequence Model.
    Provides parameter-efficient gating dynamics on sequence [batch_size, seq_len, num_features].
    """
    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout_rate=0.3):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]
        gru_out, h_n = self.gru(x)
        # Use final time-step hidden state representation
        last_hidden = gru_out[:, -1, :] # Shape: [batch_size, hidden_dim]
        return self.fc(last_hidden)


def get_model(model_name, num_features, seq_len=14):
    """
    Factory function to instantiate PyTorch models by name.
    """
    name = model_name.upper()
    if name == "MLP":
        input_dim = seq_len * num_features
        return MLPModel(input_dim=input_dim)
    elif name in ["1D-CNN", "1DCNN", "CNN1D"]:
        return CNN1DModel(num_features=num_features, seq_len=seq_len)
    elif name == "LSTM":
        return LSTMModel(num_features=num_features)
    elif name == "GRU":
        return GRUModel(num_features=num_features)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from ['MLP', '1D-CNN', 'LSTM', 'GRU'].")
