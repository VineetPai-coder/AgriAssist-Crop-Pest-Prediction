import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (
    generate_agriassist_dataset,
    load_and_preprocess_data,
    split_temporal_data,
    get_dataloaders
)
from src.trainer import train_and_benchmark_models, run_ndvi_ablation_study
from src.visualize import (
    plot_eda_time_series,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_benchmark_comparison,
    plot_ndvi_ablation
)

def main():
    print("=" * 80)
    print("      AgriAssist: Multimodal Time-Series-Based Crop Pest Outbreak Prediction      ")
    print("=" * 80)

    csv_path = "data/agriassist_multiyear_dataset.csv"
    
    # 1. Dataset Acquisition & Generation
    print("\n[Phase 1] Dataset Acquisition & Exploratory Data Analysis (EDA)...")
    generate_agriassist_dataset(output_path=csv_path, num_days=1150, seed=42)
    
    # Load dataset for EDA plot
    df_raw = pd.read_csv(csv_path)
    plot_eda_time_series(df_raw, output_path="outputs/eda_time_series.png")

    # 2. Preprocessing & Temporal Sliding Windowing (WITH NDVI)
    print("\n[Phase 2] Data Preprocessing, MinMax Feature Scaling & 14-Day Windowing...")
    full_data = load_and_preprocess_data(csv_path=csv_path, window_size=14, include_ndvi=True)
    full_splits = split_temporal_data(full_data["X_seq"], full_data["X_flat"], full_data["y"])
    
    num_features_full = full_data["X_seq"].shape[2]
    print(f"-> Full Feature Matrix Shape: Sequence={full_data['X_seq'].shape}, Tabular={full_data['X_flat'].shape}")
    print(f"-> Feature Names: {full_data['feature_names']}")
    print(f"-> Dataset Train Size: {len(full_splits['train']['y'])}, Val Size: {len(full_splits['val']['y'])}, Test Size: {len(full_splits['test']['y'])}")

    # Prepare DataLoaders for all models
    model_names = ["MLP", "1D-CNN", "LSTM", "GRU"]
    train_val_test_loaders = {}
    for name in model_names:
        is_flattened = (name == "MLP")
        tr, val, te = get_dataloaders(full_splits, batch_size=32, is_flattened=is_flattened)
        train_val_test_loaders[name] = {"train": tr, "val": val, "test": te}

    # 3. Model Implementation & Baseline Training
    print("\n[Phase 3] Training & Benchmarking Deep Learning Architectures...")
    benchmark_results = train_and_benchmark_models(
        train_val_test_loaders,
        num_features=num_features_full,
        seq_len=14,
        epochs=45
    )

    # 4. Satellite NDVI Ablation Study
    print("\n[Phase 4] Running Objective 2: Satellite NDVI Proxy Ablation Study...")
    no_ndvi_data = load_and_preprocess_data(csv_path=csv_path, window_size=14, include_ndvi=False)
    no_ndvi_splits = split_temporal_data(no_ndvi_data["X_seq"], no_ndvi_data["X_flat"], no_ndvi_data["y"])
    num_features_no_ndvi = no_ndvi_data["X_seq"].shape[2]

    ablation_results = run_ndvi_ablation_study(
        full_splits,
        no_ndvi_splits,
        num_features_full=num_features_full,
        num_features_no_ndvi=num_features_no_ndvi
    )

    # 5. Visualizations & Figure Generation
    print("\n[Phase 5] Rendering & Saving Visual Deliverable Artifacts...")
    plot_confusion_matrices(benchmark_results, output_path="outputs/confusion_matrices.png")
    plot_roc_curves(benchmark_results, output_path="outputs/roc_curves.png")
    plot_benchmark_comparison(benchmark_results, output_path="outputs/model_benchmark_comparison.png")
    plot_ndvi_ablation(ablation_results, output_path="outputs/ndvi_ablation_study.png")

    # 6. Print Consolidated Summary Table
    print("\n" + "=" * 80)
    print("                       FINAL BENCHMARK COMPARISON TABLE                        ")
    print("=" * 80)
    header = f"{'Model':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}"
    print(header)
    print("-" * len(header))

    for name in model_names:
        m = benchmark_results[name]["metrics"]
        print(f"{name:<12} | {m['Accuracy']:<10.4f} | {m['Precision']:<10.4f} | {m['Recall']:<10.4f} | {m['F1-Score']:<10.4f} | {m['ROC-AUC']:<10.4f}")
    
    print("-" * len(header))

    print("\n" + "=" * 80)
    print("                 NDVI SATELLITE PROXY ABLATION COMPARISON                     ")
    print("=" * 80)
    ablation_header = f"{'Model':<12} | {'F1 (w/ NDVI)':<14} | {'F1 (w/o NDVI)':<14} | {'AUC (w/ NDVI)':<14} | {'AUC (w/o NDVI)':<14}"
    print(ablation_header)
    print("-" * len(ablation_header))

    for name in model_names:
        w_f1 = ablation_results[name]["with_ndvi"]["F1-Score"]
        wo_f1 = ablation_results[name]["without_ndvi"]["F1-Score"]
        w_auc = ablation_results[name]["with_ndvi"]["ROC-AUC"]
        wo_auc = ablation_results[name]["without_ndvi"]["ROC-AUC"]
        print(f"{name:<12} | {w_f1:<14.4f} | {wo_f1:<14.4f} | {w_auc:<14.4f} | {wo_auc:<14.4f}")

    print("-" * len(ablation_header))
    print("\n[AgriAssist Execution Complete] All models successfully benchmarked and output figures saved to 'outputs/'.\n")

if __name__ == "__main__":
    main()
