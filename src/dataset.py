import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def generate_agriassist_dataset(output_path="data/agriassist_multiyear_dataset.csv", num_days=1150, seed=42):
    """
    Generates a realistic longitudinal daily agro-meteorological dataset spanning ~3+ years.
    Features include continuous weather parameters, NDVI reflectance, categorical crop growth stages,
    and binary pest outbreak labels.
    """
    np.random.seed(seed)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    date_range = pd.date_range(start="2023-01-01", periods=num_days, freq="D")
    day_of_year = date_range.dayofyear.values

    # Seasonal temperature signal (sine wave + noise)
    base_temp = 25 + 7 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    temperature = base_temp + np.random.normal(0, 2.0, num_days)
    temperature = np.clip(temperature, 12.0, 42.0)

    # Precipitation: sporadic rain events, higher in monsoon period (days 150 to 270)
    monsoon_prob = np.where((day_of_year >= 150) & (day_of_year <= 270), 0.45, 0.12)
    rain_events = np.random.binomial(1, monsoon_prob)
    precipitation = rain_events * np.random.exponential(scale=18.0, size=num_days)
    precipitation = np.round(precipitation, 2)

    # Relative humidity: correlated with precipitation and inversely with temp
    base_humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year - 120) / 365.25)
    humidity_rain_boost = np.where(precipitation > 5, 18.0, 0.0)
    humidity = base_humidity + humidity_rain_boost + np.random.normal(0, 4.0, num_days)
    humidity = np.clip(humidity, 30.0, 98.0)

    # Crop Growth Stage cycles (~90-100 day crop cycles repeated over years)
    crop_cycle_day = day_of_year % 90
    growth_stages = []
    for cd in crop_cycle_day:
        if cd < 20:
            growth_stages.append("early")
        elif cd < 45:
            growth_stages.append("mid")
        elif cd < 75:
            growth_stages.append("flowering")
        else:
            growth_stages.append("harvest")

    # Satellite NDVI vegetation proxy (rises during mid/flowering, drops at harvest)
    ndvi_base = []
    for cd in crop_cycle_day:
        if cd < 20:
            val = 0.20 + (cd / 20.0) * 0.25
        elif cd < 45:
            val = 0.45 + ((cd - 20) / 25.0) * 0.35
        elif cd < 75:
            val = 0.80 - ((cd - 45) / 30.0) * 0.15
        else:
            val = 0.65 - ((cd - 75) / 15.0) * 0.40
        ndvi_base.append(val)
    ndvi = np.array(ndvi_base) + np.random.normal(0, 0.03, num_days)
    ndvi = np.clip(ndvi, 0.10, 0.90)

    # Pest outbreak binary target (0 or 1)
    # Risk factors: high humidity (>72%), warm temp (22-32°C), flowering/mid stage, rain presence
    outbreak_risk = (
        0.35 * (humidity > 72).astype(float) +
        0.25 * ((temperature >= 22) & (temperature <= 32)).astype(float) +
        0.20 * np.array([1.0 if s in ["mid", "flowering"] else 0.2 for s in growth_stages]) +
        0.20 * (precipitation > 2.0).astype(float)
    )

    # Add temporal persistence / smooth risk over 3-day moving window
    outbreak_prob = pd.Series(outbreak_risk).rolling(window=3, min_periods=1).mean().values
    outbreak_prob = 1.0 / (1.0 + np.exp(-6 * (outbreak_prob - 0.65))) # Sigmoid curve thresholding
    pest_outbreak = (np.random.uniform(0, 1, num_days) < outbreak_prob).astype(int)

    df = pd.DataFrame({
        "date": date_range.strftime("%Y-%m-%d"),
        "temperature": np.round(temperature, 2),
        "relative_humidity": np.round(humidity, 2),
        "precipitation": np.round(precipitation, 2),
        "ndvi": np.round(ndvi, 4),
        "crop_growth_stage": growth_stages,
        "pest_outbreak": pest_outbreak
    })

    df.to_csv(output_path, index=False)
    print(f"[Dataset Generator] Successfully generated dataset with {len(df)} samples at '{output_path}'.")
    return df


class MinMaxScalerCustom:
    """Custom MinMax Scaler to avoid external dependencies."""
    def __init__(self):
        self.min = None
        self.max = None

    def fit_transform(self, data):
        data = np.array(data, dtype=np.float32)
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        # Prevent divide by zero
        denom = np.where((self.max - self.min) == 0, 1.0, (self.max - self.min))
        return (data - self.min) / denom

    def transform(self, data):
        data = np.array(data, dtype=np.float32)
        denom = np.where((self.max - self.min) == 0, 1.0, (self.max - self.min))
        return (data - self.min) / denom


def load_and_preprocess_data(csv_path="data/agriassist_multiyear_dataset.csv", window_size=14, include_ndvi=True):
    """
    Loads dataset, cleans, applies one-hot encoding, scales features, and constructs
    14-day sliding window tensors for sequence modeling and tabular flattening for MLP.
    """
    if not os.path.exists(csv_path):
        generate_agriassist_dataset(csv_path)

    df = pd.read_csv(csv_path)

    # Extract numerical features
    num_cols = ["temperature", "relative_humidity", "precipitation"]
    if include_ndvi:
        num_cols.append("ndvi")

    # One-hot encode crop_growth_stage
    stage_dummies = pd.get_dummies(df["crop_growth_stage"], prefix="stage").astype(float)
    
    # Combine numerical features and stage dummies
    feature_df = pd.concat([df[num_cols], stage_dummies], axis=1)
    feature_names = list(feature_df.columns)
    
    scaler = MinMaxScalerCustom()
    scaled_features = scaler.fit_transform(feature_df.values)
    labels = df["pest_outbreak"].values

    # Construct 14-day sliding window sequences
    X_seq = []
    y_seq = []
    
    # Sliding window: inputs are days t to t+window_size-1, target is label at day t+window_size-1
    for i in range(len(scaled_features) - window_size + 1):
        window = scaled_features[i : i + window_size]
        target = labels[i + window_size - 1]
        X_seq.append(window)
        y_seq.append(target)

    X_seq = np.array(X_seq, dtype=np.float32) # Shape: [N, 14, num_features]
    y_seq = np.array(y_seq, dtype=np.float32) # Shape: [N]

    # Flattened representation for MLP Baseline
    X_flat = X_seq.reshape(X_seq.shape[0], -1) # Shape: [N, 14 * num_features]

    return {
        "X_seq": X_seq,
        "X_flat": X_flat,
        "y": y_seq,
        "feature_names": feature_names,
        "scaler": scaler,
        "df_raw": df
    }


def split_temporal_data(X_seq, X_flat, y, train_ratio=0.70, val_ratio=0.15):
    """
    Splits temporal data chronologically into train, validation, and test sets.
    """
    n = len(y)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    data_splits = {
        "train": {
            "X_seq": X_seq[:train_end],
            "X_flat": X_flat[:train_end],
            "y": y[:train_end]
        },
        "val": {
            "X_seq": X_seq[train_end:val_end],
            "X_flat": X_flat[train_end:val_end],
            "y": y[train_end:val_end]
        },
        "test": {
            "X_seq": X_seq[val_end:],
            "X_flat": X_flat[val_end:],
            "y": y[val_end:]
        }
    }
    return data_splits


class AgriAssistDataset(Dataset):
    """PyTorch Dataset wrapper for sequential and tabular data."""
    def __init__(self, X, y, is_flattened=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Shape: [N, 1]
        self.is_flattened = is_flattened

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_splits, batch_size=32, is_flattened=False):
    """Creates PyTorch DataLoaders for train, val, and test splits."""
    key = "X_flat" if is_flattened else "X_seq"
    
    train_ds = AgriAssistDataset(data_splits["train"][key], data_splits["train"]["y"], is_flattened)
    val_ds = AgriAssistDataset(data_splits["val"][key], data_splits["val"]["y"], is_flattened)
    test_ds = AgriAssistDataset(data_splits["test"][key], data_splits["test"]["y"], is_flattened)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
