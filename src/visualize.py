import os
import matplotlib.pyplot as plt
import numpy as np

# Set clean matplotlib visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 1.0

def plot_eda_time_series(df_raw, output_path="outputs/eda_time_series.png"):
    """
    Plots daily meteorological time-series, satellite NDVI, and pest outbreak events.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sample = df_raw.iloc[:365].copy() # Plot first year for clarity

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # 1. Temperature & Humidity
    ax1 = axes[0]
    ax1.plot(df_sample.index, df_sample["temperature"], color="#e63946", label="Temperature (°C)", linewidth=1.8)
    ax1.set_ylabel("Temp (°C)", color="#e63946", fontweight="bold")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df_sample.index, df_sample["relative_humidity"], color="#457b9d", label="Humidity (%)", linewidth=1.5, linestyle="--")
    ax1_twin.set_ylabel("Humidity (%)", color="#457b9d", fontweight="bold")
    ax1.set_title("AgriAssist: 1-Year Agro-Meteorological & Satellite NDVI Signals", fontsize=14, fontweight="bold", pad=12)

    # 2. Precipitation
    ax2 = axes[1]
    ax2.bar(df_sample.index, df_sample["precipitation"], color="#1d3557", alpha=0.7, width=1.0)
    ax2.set_ylabel("Precip (mm)", fontweight="bold")

    # 3. Satellite NDVI Index
    ax3 = axes[2]
    ax3.plot(df_sample.index, df_sample["ndvi"], color="#2a9d8f", linewidth=2.0)
    ax3.set_ylabel("NDVI Index", fontweight="bold")

    # 4. Binary Pest Outbreak Ground Truth
    ax4 = axes[3]
    ax4.step(df_sample.index, df_sample["pest_outbreak"], color="#d62828", where="post", linewidth=2.0)
    ax4.fill_between(df_sample.index, 0, df_sample["pest_outbreak"], color="#d62828", alpha=0.3)
    ax4.set_ylabel("Pest Outbreak", fontweight="bold")
    ax4.set_xlabel("Day of Year", fontweight="bold")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["No Outbreak", "Outbreak"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved EDA time series plot to '{output_path}'.")


def plot_confusion_matrices(benchmark_results, output_path="outputs/confusion_matrices.png"):
    """
    Renders 2x2 grid of annotated confusion matrices for all 4 models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    model_names = ["MLP", "1D-CNN", "LSTM", "GRU"]
    
    cmap = plt.cm.Blues

    for idx, name in enumerate(model_names):
        ax = axes[idx // 2, idx % 2]
        cm_dict = benchmark_results[name]["metrics"]["Confusion_Matrix"]
        cm_matrix = np.array([
            [cm_dict["TN"], cm_dict["FP"]],
            [cm_dict["FN"], cm_dict["TP"]]
        ])

        im = ax.imshow(cm_matrix, interpolation='nearest', cmap=cmap)
        ax.set_title(f"{name} Model", fontsize=12, fontweight="bold")
        
        # Annotate cell numbers
        for i in range(2):
            for j in range(2):
                val = cm_matrix[i, j]
                color = "white" if val > cm_matrix.max() / 2 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"], fontweight="bold")
        ax.set_yticklabels(["True 0", "True 1"], fontweight="bold")

    fig.suptitle("Confusion Matrices Across Deep Learning Architectures", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved Confusion Matrices grid to '{output_path}'.")


def plot_roc_curves(benchmark_results, output_path="outputs/roc_curves.png"):
    """
    Superimposes ROC curves for MLP, 1D-CNN, LSTM, and GRU models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(9, 7))

    colors = {
        "MLP": "#e76f51",
        "1D-CNN": "#2a9d8f",
        "LSTM": "#264653",
        "GRU": "#e63946"
    }

    for name in ["MLP", "1D-CNN", "LSTM", "GRU"]:
        metrics = benchmark_results[name]["metrics"]
        fpr = metrics["ROC_FPR"]
        tpr = metrics["ROC_TPR"]
        auc_val = metrics["ROC-AUC"]

        plt.plot(fpr, tpr, color=colors[name], linewidth=2.2, label=f"{name} (AUC = {auc_val:.3f})")

    # Baseline random classifier line
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1.5, label="Random Guess (AUC = 0.500)")

    plt.title("AgriAssist: Comparative ROC-AUC Curves", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved ROC curves plot to '{output_path}'.")


def plot_benchmark_comparison(benchmark_results, output_path="outputs/model_benchmark_comparison.png"):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = ["MLP", "1D-CNN", "LSTM", "GRU"]
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    
    x = np.arange(len(metric_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = ["#e76f51", "#2a9d8f", "#264653", "#e63946"]

    for idx, model_name in enumerate(models):
        scores = [benchmark_results[model_name]["metrics"][k] for k in metric_keys]
        offset = x + (idx - 1.5) * width
        rects = ax.bar(offset, scores, width, label=model_name, color=colors[idx], edgecolor="white")
        
        # Annotate values on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    ax.set_ylabel("Score", fontweight="bold", fontsize=12)
    ax.set_title("Benchmark Performance Across Model Architectures", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontweight="bold", fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.legend(loc="upper right", fontsize=11, frameon=True)
    ax.grid(axis='y', linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved model benchmark comparison plot to '{output_path}'.")


def plot_ndvi_ablation(ablation_results, output_path="outputs/ndvi_ablation_study.png"):
    """
    Bar chart illustrating F1-Score and ROC-AUC impact of adding satellite NDVI proxies.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    models = ["MLP", "1D-CNN", "LSTM", "GRU"]
    
    f1_with = [ablation_results[m]["with_ndvi"]["F1-Score"] for m in models]
    f1_without = [ablation_results[m]["without_ndvi"]["F1-Score"] for m in models]

    auc_with = [ablation_results[m]["with_ndvi"]["ROC-AUC"] for m in models]
    auc_without = [ablation_results[m]["without_ndvi"]["ROC-AUC"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # F1 Score Comparison
    ax1.bar(x - width/2, f1_without, width, label="Without NDVI", color="#94d2bd")
    ax1.bar(x + width/2, f1_with, width, label="With Satellite NDVI", color="#005f73")
    ax1.set_ylabel("F1-Score", fontweight="bold")
    ax1.set_title("F1-Score Impact of Satellite NDVI Proxy", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower right")
    ax1.grid(axis='y', linestyle=":", alpha=0.6)

    # ROC-AUC Comparison
    ax2.bar(x - width/2, auc_without, width, label="Without NDVI", color="#ee9b00")
    ax2.bar(x + width/2, auc_with, width, label="With Satellite NDVI", color="#ca6702")
    ax2.set_ylabel("ROC-AUC", fontweight="bold")
    ax2.set_title("ROC-AUC Impact of Satellite NDVI Proxy", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontweight="bold")
    ax2.set_ylim([0, 1.05])
    ax2.legend(loc="lower right")
    ax2.grid(axis='y', linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Visualization] Saved NDVI ablation study plot to '{output_path}'.")
