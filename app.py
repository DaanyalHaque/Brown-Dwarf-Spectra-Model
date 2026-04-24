import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import plotly.graph_objects as go
import joblib
import gdown
import os

if not os.path.exists("smallspectra.npy"):
    gdown.download("https://drive.google.com/uc?id=1D1RtGRUe5KfMg3OnBzV2lXH8f0baUubw", "smallspectra.npy", quiet=False)

# Page config
st.set_page_config(page_title="Brown Dwarf Spectrum Predictor", layout="wide")
st.title("🌟 Brown Dwarf Spectrum Predictor")
st.markdown("Adjust the parameters below to predict the spectrum of a brown dwarf.")

# Load model and scalers
@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model("pca_model.keras")
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    pca = joblib.load("pca.pkl")
    wavelengths = np.load("wavelength.npy")
    smallparams = np.load("smallparams.npy")
    smallspectra = np.load("smallspectra.npy")
    return model, x_scaler, y_scaler, pca, wavelengths, smallparams, smallspectra

model, x_scaler, y_scaler, pca, wavelengths, smallparams, smallspectra = load_model_and_scalers()

# Sidebar sliders
st.sidebar.header("Brown Dwarf Parameters")

temperature_real = st.sidebar.slider("Temperature (K)", min_value=200, max_value=2000, value=630, step=10)
temperature = np.log10(temperature_real)

co_ratio = st.sidebar.slider("C/O Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

no_ratio_real = st.sidebar.slider("N/O Ratio", min_value=0.001, max_value=1.0, value=0.03, step=0.001)
no_ratio = np.log10(no_ratio_real)

so_ratio_real = st.sidebar.slider("S/O Ratio", min_value=0.001, max_value=1.0, value=0.03, step=0.001)
so_ratio = np.log10(so_ratio_real)

metallicity = st.sidebar.slider("Metallicity", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
kzz = st.sidebar.slider("Kzz", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
gravity = st.sidebar.slider("Gravity (log g)", min_value=3.0, max_value=6.0, value=4.5, step=0.1)

# Predict
params = np.array([[temperature, co_ratio, no_ratio, so_ratio, metallicity, kzz, gravity]])
params_scaled = x_scaler.transform(params)
pred_pca = model.predict(params_scaled, verbose=0)
pred_scaled = pca.inverse_transform(pred_pca)
pred_log = y_scaler.inverse_transform(pred_scaled)
pred_flux = 10**pred_log

# Find nearest match
scaler_nn = StandardScaler()
smallparams_scaled = scaler_nn.fit_transform(smallparams)
input_scaled = scaler_nn.transform(params)
distances = cdist(input_scaled, smallparams_scaled, metric='euclidean')[0]
nearest_idx = np.argmin(distances)
nearest_flux = smallspectra[nearest_idx]

# Plot
show_nearest = st.sidebar.checkbox("Show Nearest Real Spectrum", value=False)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=wavelengths,
    y=pred_flux[0],
    mode='lines',
    name='Predicted',
    line=dict(color='royalblue', width=1.5)
))
if show_nearest:
    fig.add_trace(go.Scatter(
        x=wavelengths,
        y=nearest_flux,
        mode='lines',
        name=f'Nearest Real (index {nearest_idx})',
        line=dict(color='orange', width=1.5, dash='dash')
    ))
fig.update_layout(
    xaxis_title="Wavelength (μm)",
    yaxis_title="Flux",
    title="Predicted Brown Dwarf Spectrum",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# Show params
st.subheader("Current Parameters")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Temperature", f"{temperature_real} K")
col2.metric("C/O Ratio", f"{co_ratio:.2f}")
col3.metric("N/O Ratio", f"{no_ratio_real:.4f}")
col4.metric("S/O Ratio", f"{so_ratio_real:.4f}")
col1.metric("Metallicity", f"{metallicity:.1f}")
col2.metric("Kzz", f"{kzz:.1f}")
col3.metric("Gravity", f"{gravity:.1f}")

if show_nearest:
    st.caption(f"Nearest match: Brown Dwarf index {nearest_idx} (distance: {distances[nearest_idx]:.4f})")
