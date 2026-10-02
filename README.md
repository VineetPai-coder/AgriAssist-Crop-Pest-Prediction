# 🌾 AgriAssist: Multimodal Time-Series-Based Crop Pest Outbreak Prediction

> **ICT-4442 Deep Learning Mini Project** | Manipal Institute of Technology, MAHE  
> **Phase 2 Interim Submission** — September 2026

---

## 📋 Project Overview

**AgriAssist** is an end-to-end deep learning framework for early prediction of crop pest outbreak risks by fusing:
- 🌡️ Daily agro-meteorological records (temperature, humidity, precipitation)
- 🛰️ Satellite vegetation health proxies (NDVI reflectance index)
- 🌱 Crop phenological growth stages (early, mid, flowering, harvest)

Using a **14-day sliding temporal window**, four deep learning architectures are benchmarked for binary outbreak classification.

---

## 👥 Team Members

| Name | Registration No. | Model / Role |
|:---|:---:|:---|
| Nihar P Shetty | 230911027 | Data Preprocessing & MLP Baseline |
| Pranav Prabhu | 230911414 | 14-Day Windowing & LSTM Model |
| Vineet Pai | 230911450 | EDA, 1D-CNN Model & Metrics |
| Sunku Venkata Subhash | 230911370 | GRU Model & NDVI Ablation Study |

---

## 🎯 Objectives

1. **Comparative Benchmarking**: Train and evaluate four architectures — MLP, 1D-CNN, LSTM, and GRU — on a standardized temporal evaluation protocol.
2. **NDVI Proxy Impact (Ablation Study)**: Quantify the predictive improvement from integrating satellite NDVI vegetation indices over meteorological features alone.

---

## 🏗️ Project Structure

```
AgriAssist/
├── main.py                     # End-to-end training & evaluation pipeline
├── data/
│   └── agriassist_multiyear_dataset.csv   # 3+ year longitudinal agro-weather dataset
├── src/
│   ├── dataset.py              # Data loading, scaling, windowing, DataLoaders
│   ├── models.py               # MLP, 1D-CNN, LSTM, GRU model definitions
│   ├── metrics.py              # Accuracy, Precision, Recall, F1, ROC-AUC
│   ├── trainer.py              # Training loop, early stopping, ablation engine
│   └── visualize.py            # EDA, confusion matrices, ROC curves, plots
└── outputs/
    ├── eda_time_series.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── model_benchmark_comparison.png
    └── ndvi_ablation_study.png
```

---

## 🧠 Model Architectures

| Model | Input | Key Design |
|:---|:---|:---|
| **MLP Baseline** | Flattened `[N, 112]` window | Linear layers + BatchNorm + Dropout |
| **1D-CNN** | Sequence `[N, 14, 8]` | Conv1D layers + AdaptiveAvgPool |
| **LSTM** | Sequence `[N, 14, 8]` | 2-layer LSTM, last hidden state |
| **GRU** | Sequence `[N, 14, 8]` | 2-layer GRU, last hidden state |

---

## 📊 Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| MLP Baseline | 0.8187 | 0.4706 | 0.2667 | 0.3404 | 0.6870 |
| 1D-CNN Variant | 0.8187 | 0.4737 | 0.3000 | 0.3673 | 0.7017 |
| LSTM Sequence Model | **0.8421** | **0.6667** | 0.2000 | 0.3077 | **0.7069** |
| GRU Sequence Model | 0.8246 | 0.5000 | **0.3000** | **0.3750** | 0.7054 |

---

## 🛰️ NDVI Ablation Study (Objective 2)

| Model | F1 (With NDVI) | F1 (Without NDVI) | AUC (With NDVI) | AUC (Without NDVI) |
|:---|:---:|:---:|:---:|:---:|
| MLP Baseline | **0.3721** | 0.3636 | 0.6742 | 0.6820 |
| 1D-CNN Variant | 0.3721 | 0.4091 | 0.6879 | 0.7180 |
| LSTM Sequence Model | **0.3500** | 0.3077 | 0.7005 | 0.7061 |
| GRU Sequence Model | **0.3500** | 0.3077 | **0.7104** | 0.6913 |

---

## ⚙️ Setup & Usage

### Prerequisites
```bash
pip install torch torchvision numpy pandas matplotlib
```

### Run the Full Pipeline
```bash
python main.py
```

This will:
1. Generate the 3-year agro-meteorological dataset
2. Preprocess, scale, and apply 14-day sliding windows
3. Train and benchmark all 4 models (MLP, 1D-CNN, LSTM, GRU)
4. Run the NDVI ablation study
5. Save all result plots to `outputs/`

---

## 📈 Visual Outputs

| Plot | Description |
|:---|:---|
| `eda_time_series.png` | 1-year agro-meteorological & NDVI signals timeline |
| `confusion_matrices.png` | 2×2 grid of confusion matrices across all 4 models |
| `roc_curves.png` | Superimposed ROC-AUC curves |
| `model_benchmark_comparison.png` | Grouped bar chart of all metrics |
| `ndvi_ablation_study.png` | F1 & AUC comparison with/without NDVI |

---

## 🏫 Institution

**Department of Information & Communication Technology**  
School of Computer Engineering  
Manipal Institute of Technology, Manipal Academy of Higher Education (MAHE)
