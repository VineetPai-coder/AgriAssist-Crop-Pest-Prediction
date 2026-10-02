import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models import get_model
from src.metrics import calculate_metrics

class Trainer:
    """
    Handles model training, early stopping, evaluation, and checkpoint management.
    """
    def __init__(self, model, lr=1e-3, weight_decay=1e-4, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                running_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                all_probs.extend(probs)
                all_targets.extend(y_batch.cpu().numpy())

        epoch_loss = running_loss / len(dataloader.dataset)
        all_probs = np.array(all_probs).ravel()
        all_targets = np.array(all_targets).ravel()
        
        metrics = calculate_metrics(all_targets, all_probs)
        metrics["Loss"] = epoch_loss
        return metrics, all_probs, all_targets

    def fit(self, train_loader, val_loader, epochs=40, patience=10):
        best_val_loss = float("inf")
        best_model_weights = None
        history = {"train_loss": [], "val_loss": [], "val_f1": []}

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics, _, _ = self.evaluate(val_loader)
            val_loss = val_metrics["Loss"]
            val_f1 = val_metrics["F1-Score"]

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # Early stopping
                break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        return history


def train_and_benchmark_models(train_val_test_loaders, num_features, seq_len=14, epochs=45):
    """
    Trains and evaluates all 4 benchmark models: MLP, 1D-CNN, LSTM, GRU.
    """
    model_names = ["MLP", "1D-CNN", "LSTM", "GRU"]
    results = {}

    for name in model_names:
        is_flattened = (name == "MLP")
        train_loader = train_val_test_loaders[name]["train"]
        val_loader = train_val_test_loaders[name]["val"]
        test_loader = train_val_test_loaders[name]["test"]

        model = get_model(name, num_features=num_features, seq_len=seq_len)
        trainer = Trainer(model, lr=1e-3 if name != "MLP" else 2e-3)

        print(f"[Trainer] Training {name} Model...")
        history = trainer.fit(train_loader, val_loader, epochs=epochs, patience=12)
        test_metrics, test_probs, test_targets = trainer.evaluate(test_loader)

        results[name] = {
            "metrics": test_metrics,
            "probs": test_probs,
            "targets": test_targets,
            "history": history,
            "model": model
        }
        print(f"[Trainer] {name} Test Metrics -> F1: {test_metrics['F1-Score']:.4f} | ROC-AUC: {test_metrics['ROC-AUC']:.4f} | Accuracy: {test_metrics['Accuracy']:.4f}")

    return results


def run_ndvi_ablation_study(full_data_splits, data_splits_no_ndvi, num_features_full, num_features_no_ndvi):
    """
    Quantifies the predictive impact of integrating satellite vegetation health proxies (NDVI).
    Compares models trained WITH vs WITHOUT NDVI.
    """
    from src.dataset import get_dataloaders
    print("\n[Ablation Study] Running NDVI Proxy Impact Evaluation...")

    model_names = ["MLP", "1D-CNN", "LSTM", "GRU"]
    ablation_results = {}

    for name in model_names:
        is_flattened = (name == "MLP")
        
        # Loaders WITH NDVI
        tr_full, val_full, te_full = get_dataloaders(full_data_splits, batch_size=32, is_flattened=is_flattened)
        model_full = get_model(name, num_features=num_features_full)
        trainer_full = Trainer(model_full)
        trainer_full.fit(tr_full, val_full, epochs=35, patience=10)
        metrics_full, _, _ = trainer_full.evaluate(te_full)

        # Loaders WITHOUT NDVI
        tr_no, val_no, te_no = get_dataloaders(data_splits_no_ndvi, batch_size=32, is_flattened=is_flattened)
        model_no = get_model(name, num_features=num_features_no_ndvi)
        trainer_no = Trainer(model_no)
        trainer_no.fit(tr_no, val_no, epochs=35, patience=10)
        metrics_no, _, _ = trainer_no.evaluate(te_no)

        ablation_results[name] = {
            "with_ndvi": metrics_full,
            "without_ndvi": metrics_no
        }
    return ablation_results
