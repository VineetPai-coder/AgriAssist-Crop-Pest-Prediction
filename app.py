import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

# Custom local module imports
from src.dataset import generate_agriassist_dataset, load_and_preprocess_data
from src.models import get_model

st.set_page_config(
    page_title="AgriAssist Outbreak Prediction Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌾 AgriAssist: Multimodal Crop Pest Outbreak Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ICT-4442 Deep Learning Mini Project | Manipal Institute of Technology</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Benchmarking", 
    "🛰️ NDVI Ablation Study", 
    "📈 Dataset & EDA", 
    "🔮 Live Outbreak Predictor"
])

# -------------------------------------------------------------
# TAB 1: MODEL BENCHMARKING
# -------------------------------------------------------------
with tab1:
    st.subheader("Deep Learning Architecture Comparison")
    st.write("Performance evaluation of all 4 models on the common 171-window test holdout:")
    
    col_bench1, col_bench2 = st.columns([1.2, 1])
    
    with col_bench1:
        benchmark_data = {
            "Model": ["MLP (Baseline)", "1D-CNN", "LSTM", "GRU"],
            "Architecture Family": ["Feedforward Tabular", "Temporal Convolutional", "Recurrent (Memory Cell)", "Recurrent (Gated Unit)"],
            "Accuracy": [0.8187, 0.8187, 0.8421, 0.8246],
            "Precision": [0.4706, 0.4737, 0.6667, 0.5000],
            "Recall": [0.2667, 0.3000, 0.2000, 0.3000],
            "F1-Score": [0.3404, 0.3673, 0.3077, 0.3750],
            "ROC-AUC": [0.6870, 0.7017, 0.7069, 0.7054]
        }
        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#c8e6c9"), use_container_width=True)
        
        st.info("💡 **Key Takeaways:**\n"
                "- **Best Accuracy & ROC-AUC:** **LSTM** (84.21% Acc, 0.7069 AUC) captures stable temporal sequences with fewer false alarms.\n"
                "- **Best Outbreak Detection (F1):** **GRU** (0.3750 F1) balances positive pest outbreak recall with reasonable precision.")

    with col_bench2:
        if os.path.exists("outputs/model_benchmark_comparison.png"):
            st.image("outputs/model_benchmark_comparison.png", caption="Model Benchmark Comparison across Metrics", use_container_width=True)
        else:
            st.warning("Benchmark image not found. Run main.py first.")

    st.divider()
    st.subheader("Diagnostic Evaluation: Confusion Matrices & ROC Curves")
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("outputs/confusion_matrices.png"):
            st.image("outputs/confusion_matrices.png", caption="Confusion Matrices across 4 Models (Threshold 0.5)", use_container_width=True)
    with c2:
        if os.path.exists("outputs/roc_curves.png"):
            st.image("outputs/roc_curves.png", caption="Superimposed ROC-AUC Curves", use_container_width=True)

# -------------------------------------------------------------
# TAB 2: NDVI ABLATION STUDY
# -------------------------------------------------------------
with tab2:
    st.subheader("Objective 2: Impact of Satellite NDVI Vegetation Index")
    st.write("Evaluating predictive gain from fusing satellite vegetation health proxy (NDVI) with meteorological covariates:")
    
    col_ab1, col_ab2 = st.columns([1.2, 1])
    with col_ab1:
        ablation_data = {
            "Model": ["MLP", "1D-CNN", "LSTM", "GRU"],
            "F1 (With NDVI)": [0.3721, 0.3721, 0.3500, 0.3500],
            "F1 (Without NDVI)": [0.3636, 0.4091, 0.3077, 0.3077],
            "Δ F1 Impact": ["+0.0085", "-0.0370", "+0.0423", "+0.0423"],
            "AUC (With NDVI)": [0.6742, 0.6879, 0.7005, 0.7104],
            "AUC (Without NDVI)": [0.6820, 0.7180, 0.7061, 0.6913],
            "Δ AUC Impact": ["-0.0078", "-0.0301", "-0.0056", "+0.0191"]
        }
        df_ab = pd.DataFrame(ablation_data)
        st.dataframe(df_ab, use_container_width=True)
        
        st.markdown("""
        **Ablation Insights:**
        - **Recurrent Architectures (LSTM & GRU):** Exhibit consistent **+0.0423 gain in F1-score** when NDVI is incorporated, confirming that vegetation health trajectories offer complementary signals over weather alone.
        - **GRU:** Shows improvement across both F1 (+0.0423) and ROC-AUC (+0.0191).
        """)

    with col_ab2:
        if os.path.exists("outputs/ndvi_ablation_study.png"):
            st.image("outputs/ndvi_ablation_study.png", caption="NDVI Ablation Impact (F1 & ROC-AUC)", use_container_width=True)

# -------------------------------------------------------------
# TAB 3: DATASET & EDA
# -------------------------------------------------------------
with tab3:
    st.subheader("Agro-Meteorological Longitudinal Dataset")
    if os.path.exists("data/agriassist_multiyear_dataset.csv"):
        df_data = pd.read_csv("data/agriassist_multiyear_dataset.csv")
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Total Days Recorded", f"{len(df_data):,} days")
        col_d2.metric("Date Span", f"{df_data['date'].iloc[0]} to {df_data['date'].iloc[-1]}")
        outbreak_rate = (df_data['pest_outbreak'].sum() / len(df_data)) * 100
        col_d3.metric("Outbreak Prevalence", f"{outbreak_rate:.1f}% ({df_data['pest_outbreak'].sum()} days)")
        col_d4.metric("Window Size", "14 Days Sliding")
        
        st.dataframe(df_data.head(15), use_container_width=True)
    else:
        st.warning("Dataset CSV not found. Please run main.py to generate it.")
        
    st.divider()
    st.subheader("Exploratory Data Analysis: Multi-Signal Timeline")
    if os.path.exists("outputs/eda_time_series.png"):
        st.image("outputs/eda_time_series.png", caption="1-Year Longitudinal Agro-Meteorological & NDVI Dynamics", use_container_width=True)

# -------------------------------------------------------------
# TAB 4: LIVE OUTBREAK PREDICTOR
# -------------------------------------------------------------
with tab4:
    st.subheader("Interactive 14-Day Simulation & Outbreak Risk Estimator")
    st.write("Simulate recent agricultural conditions over a 14-day window to evaluate live predictions across models:")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        st.markdown("**Weather Conditions (Mean)**")
        avg_temp = st.slider("Mean Temperature (°C)", 10.0, 45.0, 28.5, 0.5)
        avg_hum = st.slider("Relative Humidity (%)", 20.0, 100.0, 82.0, 1.0)
        avg_rain = st.slider("Daily Precipitation (mm)", 0.0, 60.0, 15.0, 0.5)

    with col_in2:
        st.markdown("**Vegetation & Crop Dynamics**")
        ndvi_val = st.slider("Satellite NDVI Index (0.1 to 0.9)", 0.10, 0.90, 0.72, 0.02)
        crop_stage = st.selectbox("Current Crop Phenological Stage", ["early", "mid", "flowering", "harvest"], index=2)

    with col_in3:
        st.markdown("**Outbreak Risk Factors**")
        temp_risk = 1.0 if 22.0 <= avg_temp <= 32.0 else 0.0
        hum_risk = 1.0 if avg_hum > 72.0 else 0.0
        rain_risk = 1.0 if avg_rain > 2.0 else 0.0
        stage_risk = 1.0 if crop_stage in ["mid", "flowering"] else 0.2
        
        composite_risk = (0.35 * hum_risk + 0.25 * temp_risk + 0.20 * stage_risk + 0.20 * rain_risk)
        st.write(f"- Optimal Pest Temperature (22-32°C): **{'Yes' if temp_risk else 'No'}**")
        st.write(f"- High Humidity (>72%): **{'Yes' if hum_risk else 'No'}**")
        st.write(f"- Wet Foliage (>2mm rain): **{'Yes' if rain_risk else 'No'}**")
        st.write(f"- Vulnerable Crop Stage: **{'Yes' if stage_risk == 1.0 else 'Moderate'}**")
        
        prob_est = 1.0 / (1.0 + np.exp(-6.0 * (composite_risk - 0.65)))
        st.metric("Estimated Outbreak Risk Score", f"{prob_est*100:.1f}%")

    if st.button("🚀 Run Live Inference Across All 4 Models", type="primary"):
        # Synthesize a 14-day window around the selected parameters
        t_seq = np.linspace(avg_temp - 2, avg_temp + 1, 14)
        h_seq = np.linspace(avg_hum - 5, avg_hum + 3, 14)
        r_seq = np.linspace(avg_rain * 0.7, avg_rain * 1.2, 14)
        n_seq = np.linspace(ndvi_val - 0.05, ndvi_val, 14)
        
        stage_early = [1.0 if crop_stage == 'early' else 0.0] * 14
        stage_flowering = [1.0 if crop_stage == 'flowering' else 0.0] * 14
        stage_harvest = [1.0 if crop_stage == 'harvest' else 0.0] * 14
        stage_mid = [1.0 if crop_stage == 'mid' else 0.0] * 14
        
        # MinMax normalise with reasonable empirical bounds
        t_norm = (t_seq - 12.0) / (40.0 - 12.0)
        h_norm = (h_seq - 30.0) / (100.0 - 30.0)
        r_norm = r_seq / 60.0
        n_norm = (n_seq - 0.1) / (0.9 - 0.1)
        
        seq_features = np.column_stack([
            t_norm, h_norm, r_norm, n_norm,
            stage_early, stage_flowering, stage_harvest, stage_mid
        ])  # Shape [14, 8]
        
        seq_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0)  # [1, 14, 8]
        flat_tensor = seq_tensor.view(1, -1)  # [1, 112]
        
        st.subheader("Model Prediction Results")
        res_cols = st.columns(4)
        
        models_info = [
            ("MLP", "mlp", flat_tensor, res_cols[0]),
            ("1D-CNN", "cnn1d", seq_tensor, res_cols[1]),
            ("LSTM", "lstm", seq_tensor, res_cols[2]),
            ("GRU", "gru", seq_tensor, res_cols[3])
        ]
        
        for name, mtype, inp, col in models_info:
            with col:
                model = get_model(mtype, input_dim=8, seq_len=14)
                model.eval()
                with torch.no_grad():
                    logit = model(inp).item()
                    prob = 1.0 / (1.0 + np.exp(-logit))
                    outbreak = prob >= 0.5
                
                status_color = "#d32f2f" if outbreak else "#388e3c"
                status_text = "⚠️ HIGH OUTBREAK RISK" if outbreak else "✅ LOW / SAFE RISK"
                
                st.markdown(f"""
                <div style="border: 2px solid {status_color}; border-radius: 8px; padding: 12px; text-align: center;">
                    <h4>{name}</h4>
                    <p style="font-size: 1.4rem; font-weight: bold; color: {status_color}; margin: 5px 0;">{prob*100:.1f}%</p>
                    <p style="font-weight: 600; color: {status_color};">{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
