from pathlib import Path
import io
import sys

import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import DenoisingCNN
from dncnn_model import DnCNN
from common import NoiseDetector, LABELS, REMOVE_ORDER

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Intelligent Image Denoising Suite",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN UI & CSS INJECTION ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main background & container styling */
    .main {
        background-color: #f8fafc;
    }

    /* Custom Card Containers */
    div.stMarkdown, div.stButton, div.stDownloadButton {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: #ffffff;
        padding: 16px 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
    }
    div[data-testid="stMetric"] label {
        font-weight: 600 !important;
        color: #64748b !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Sidebar Enhancement */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b;
        color: white;
        border-color: #334155;
    }

    /* Buttons */
    .stButton button, .stDownloadButton button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
        border: none;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        width: 100%;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3);
    }

    /* Success / Warning Banners */
    div.stSuccess, div.stWarning, div.stInfo {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

BASIC_CHECKPOINTS = {
    "Gaussian": {
        "Low": "denoising_gaussian_low.pth",
        "Medium": "denoising_gaussian_medium.pth",
        "High": "denoising_gaussian_high.pth",
    },
    "Salt-and-pepper": {
        "Low": "denoising_salt-and-pepper_low.pth",
        "Medium": "denoising_salt-and-pepper_medium.pth",
        "High": "denoising_salt-and-pepper_high.pth",
    },
    "Speckle": {
        "Low": "denoising_speckle_low.pth",
        "Medium": "denoising_speckle_medium.pth",
        "High": "denoising_speckle_high.pth",
    },
}

DNCNN_CHECKPOINTS = {
    "Gaussian": "denoise_gaussian.pt",
    "Salt-and-pepper": "denoise_sp.pt",
    "Speckle": "denoise_speckle.pt",
}

NOISE_LEVELS = {
    "Gaussian": {"Low": 10 / 255.0, "Medium": 25 / 255.0, "High": 50 / 255.0},
    "Salt-and-pepper": {"Low": 0.01, "Medium": 0.03, "High": 0.05},
    "Speckle": {"Low": 0.01, "Medium": 0.04, "High": 0.08},
}

DETECTOR_LABELS = {
    "gaussian": "Gaussian",
    "sp": "Salt-and-pepper",
    "speckle": "Speckle",
}

DISPLAY_LABELS = {
    "gaussian": "Gaussian",
    "sp": "Salt-and-Pepper",
    "speckle": "Speckle",
}


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def array_to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    image = Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def add_project_noise(clean: torch.Tensor, noise_type: str, severity: str, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    amount = NOISE_LEVELS[noise_type][severity]
    noisy = clean.clone()
    channels, height, width = noisy.shape

    if noise_type == "Gaussian":
        noise = torch.randn(noisy.shape, generator=generator, dtype=noisy.dtype) * amount
        noisy = torch.round(torch.clamp(noisy + noise, 0.0, 1.0) * 255.0) / 255.0
    elif noise_type == "Salt-and-pepper":
        random_values = torch.rand((height, width), generator=generator)
        pepper_mask = random_values < (amount / 2)
        salt_mask = (random_values >= amount / 2) & (random_values < amount)
        for channel in range(channels):
            noisy[channel][pepper_mask] = 0.0
            noisy[channel][salt_mask] = 1.0
    else:  # Speckle
        noise = torch.randn(noisy.shape, generator=generator, dtype=noisy.dtype) * (amount ** 0.5)
        noisy = torch.round(torch.clamp(noisy + noisy * noise, 0.0, 1.0) * 255.0) / 255.0
    return noisy


@st.cache_resource(show_spinner="Loading noise detector...")
def load_detector():
    detector_path = MODEL_DIR / "noise_detector.pt"
    if not detector_path.is_file():
        raise FileNotFoundError(f"Missing noise detector checkpoint: {detector_path}")
    detector_checkpoint = torch.load(detector_path, map_location=DEVICE, weights_only=False)

    if "labels" not in detector_checkpoint:
        raise RuntimeError("Detector checkpoint is missing label metadata.")

    saved_labels = list(detector_checkpoint["labels"])
    if saved_labels != list(LABELS):
        raise RuntimeError(f"Detector label mismatch: {saved_labels} != {LABELS}")

    detector = NoiseDetector().to(DEVICE)
    detector.load_state_dict(detector_checkpoint["model"], strict=True)
    detector.eval()
    return detector


@st.cache_resource(show_spinner="Loading selected DnCNN model...")
def load_dncnn_model(noise_type: str):
    filename = DNCNN_CHECKPOINTS[noise_type]
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing DnCNN checkpoint: {path}")
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = DnCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


@st.cache_resource(show_spinner="Loading selected Basic CNN model...")
def load_basic_model(noise_type: str, severity: str):
    filename = BASIC_CHECKPOINTS[noise_type][severity]
    path = MODEL_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing Basic CNN checkpoint: {path}")
    model = DenoisingCNN().to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def calculate_metrics(reference: np.ndarray, prediction: np.ndarray):
    prediction = np.clip(prediction, 0.0, 1.0)
    mse = float(np.mean((reference - prediction) ** 2))
    psnr = float(peak_signal_noise_ratio(reference, prediction, data_range=1.0))

    h, w = reference.shape[:2]
    min_dim = min(h, w)
    if min_dim < 3:
        ssim = float("nan")
    else:
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim = float(
            structural_similarity(
                reference, prediction, data_range=1.0, channel_axis=2, win_size=win_size
            )
        )
    return mse, psnr, ssim


def get_patch_positions(length, patch_size=128, overlap=32):
    if length <= patch_size:
        return [0]
    stride = patch_size - overlap
    positions = list(range(0, length - patch_size + 1, stride))
    last_position = length - patch_size
    if positions[-1] != last_position:
        positions.append(last_position)
    return positions


def denoise_full_resolution(
    image_tensor,
    denoising_model,
    patch_size=128,
    overlap=32,
    return_cpu=True,
):
    channels, original_height, original_width = image_tensor.shape

    pad_height = max(0, patch_size - original_height)
    pad_width = max(0, patch_size - original_width)

    padded = torch.nn.functional.pad(
        image_tensor.to(DEVICE).unsqueeze(0),
        (0, pad_width, 0, pad_height),
        mode="replicate",
    ).squeeze(0)

    _, height, width = padded.shape

    y_positions = get_patch_positions(height, patch_size, overlap)
    x_positions = get_patch_positions(width, patch_size, overlap)

    weight_1d = torch.hann_window(
        patch_size, periodic=False, device=DEVICE
    ).clamp(min=0.05)

    weight_map = torch.outer(weight_1d, weight_1d).unsqueeze(0).unsqueeze(0)

    restored_sum = torch.zeros((1, channels, height, width), device=DEVICE)
    weight_sum = torch.zeros((1, 1, height, width), device=DEVICE)

    denoising_model.eval()
    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                patch_input = padded[:, y:y + patch_size, x:x + patch_size].unsqueeze(0)
                restored_patch = denoising_model(patch_input).clamp(0, 1)
                restored_sum[:, :, y:y + patch_size, x:x + patch_size] += restored_patch * weight_map
                weight_sum[:, :, y:y + patch_size, x:x + patch_size] += weight_map

    restored = (restored_sum / weight_sum.clamp(min=1e-8))[:, :, :original_height, :original_width].squeeze(0)
    return restored.cpu() if return_cpu else restored

def detect_and_denoise(noisy_tensor, detector, threshold=0.5):
    channels, orig_h, orig_w = noisy_tensor.shape
    patch_size = 128

    pad_h = max(0, patch_size - orig_h)
    pad_w = max(0, patch_size - orig_w)

    padded_noisy = torch.nn.functional.pad(
        noisy_tensor.unsqueeze(0),
        (0, pad_w, 0, pad_h),
        mode="replicate"
    ).squeeze(0)

    _, padded_h, padded_w = padded_noisy.shape

    y_pos = get_patch_positions(padded_h, patch_size, 0)
    x_pos = get_patch_positions(padded_w, patch_size, 0)

    patches = [
        padded_noisy[:, y:y + patch_size, x:x + patch_size].unsqueeze(0)
        for y in y_pos
        for x in x_pos
    ]

    patch_logits = []
    detector.eval()
    with torch.inference_mode():
        for p in patches:
            logits = detector(p.to(DEVICE)).float()
            patch_logits.append(logits)

    mean_logits = torch.stack(patch_logits).mean(dim=0)
    probabilities = torch.sigmoid(mean_logits).squeeze(0).cpu().numpy()

    detected = {
        label: float(probabilities[LABELS.index(label)]) >= threshold
        for label in LABELS
    }
    restored_tensor = noisy_tensor.to(DEVICE)
    applied_models = []

    with torch.inference_mode():
        for label in REMOVE_ORDER:
            if detected.get(label, False):
                model_key = DETECTOR_LABELS[label]
                model_to_use = load_dncnn_model(model_key)
                restored_tensor = denoise_full_resolution(
                    restored_tensor, model_to_use, return_cpu=False
                )
                applied_models.append(label)

    return (
        restored_tensor.cpu(),
        probabilities,
        detected,
        applied_models,
    )
   



# --- RESEARCH & PROJECT RESOURCES ---
with st.sidebar:
    with st.expander("📚 Research & Sources", expanded=False):
        st.markdown("**A Noise-Adaptive Patch-Based Image Restoration Pipeline**")
        st.markdown("[📄 Read the research paper (PDF)](https://github.com/Navadeep-Bandi/image-denoising-cnn/blob/main/docs/Research_paper.pdf)")
        st.markdown("[📊 Experimental results](https://github.com/Navadeep-Bandi/image-denoising-cnn/blob/main/docs/RESULTS.md)")
        st.markdown("[🧪 Training protocol](https://github.com/Navadeep-Bandi/image-denoising-cnn/blob/main/docs/TRAINING_PROTOCOL.md)")
        st.markdown("[💻 Source code](https://github.com/Navadeep-Bandi/image-denoising-cnn/tree/main/src)")
        st.markdown("[📦 GitHub repository](https://github.com/Navadeep-Bandi/image-denoising-cnn)")

# --- UI HEADER & BADGES ---
st.markdown("<h1>✨ AI-Powered Image Denoising Suite</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748b; font-size: 1.05rem; margin-top: -10px;'>Advanced multi-label convolutional noise detection and high-fidelity artifact restoration.</p>", unsafe_allow_html=True)

# Status Pill Badge
st.markdown(f"""
<div style="display: flex; gap: 12px; margin-bottom: 24px;">
    <div style="background: #eff6ff; color: #1d4ed8; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; border: 1px solid #dbeafe;">
        ⚡ Inference Device: <span style="text-transform: uppercase;">{DEVICE}</span>
    </div>
    <div style="background: #f1f5f9; color: #334155; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; border: 1px solid #e2e8f0;">
        🎯 Architecture Target: 128 × 128 Patch Kernel
    </div>
</div>
""", unsafe_allow_html=True)

# Workflow Selection Card
st.markdown("### 🧭 Select Pipeline Mode")
mode = st.radio(
    "Choose workflow",
    ["Automatic Noise Detection", "Manual Noise Generation & Denoising"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

if mode == "Automatic Noise Detection":
    st.markdown("### 📤 Upload Target Image")
    uploaded_file = st.file_uploader(
        "Choose a noisy image file",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="automatic_upload",
    )
    
    threshold = st.slider("🔍 Detector Confidence Threshold", min_value=0.05, max_value=0.80, value=0.50, step=0.05)

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            input_tensor = image_to_tensor(original_image)
            detector = load_detector()

            with st.spinner("Analyzing multi-label noise profiles & executing sequential restoration..."):
                restored_tensor, probabilities, detected, applied = detect_and_denoise(
                    input_tensor, detector, threshold
                )

            noisy_array = tensor_to_array(input_tensor)
            restored_array = tensor_to_array(restored_tensor)

            st.markdown("---")
            st.markdown("### 📊 Noise Classification Breakdown")
            detection_cols = st.columns(3)
            for index, label in enumerate(LABELS):
                prob_idx = LABELS.index(label)
                probability = float(probabilities[prob_idx])
                with detection_cols[index]:
                    st.metric(
                        DISPLAY_LABELS[label],
                        f"{probability * 100:.1f}%",
                        delta="Detected" if detected[label] else "Clear",
                        delta_color="normal" if detected[label] else "off",
                    )
                    st.progress(min(max(probability, 0.0), 1.0))

            st.markdown("---")
            st.markdown("### 🛠️ Restoration Pipeline Status")
            if applied:
                applied_str = " → ".join(DISPLAY_LABELS[k] for k in applied)
                st.success(f"**Target Noise Identified:** {', '.join(DISPLAY_LABELS[k] for k in LABELS if detected[k])}  \n**Execution Graph:** {applied_str}")
            else:
                st.warning("⚠️ No supported noise profile crossed the activation threshold. Input passed through unmodified.")

            st.markdown("### 👁️ Visual Comparison & Difference Map")
            noisy_vis = (noisy_array * 255.0) if noisy_array.max() <= 1.0 else noisy_array
            restored_vis = (restored_array * 255.0) if restored_array.max() <= 1.0 else restored_array

            residual_map = np.abs(noisy_vis.astype(np.float32) - restored_vis.astype(np.float32))
            residual_map = np.clip(residual_map * 3.0, 0, 255).astype(np.uint8)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(noisy_vis.astype(np.uint8), caption=f"Noisy Input ({noisy_array.shape[1]}×{noisy_array.shape[0]})", use_container_width=True)
            with col2:
                st.image(restored_vis.astype(np.uint8), caption=f"Restored Output ({restored_array.shape[1]}×{restored_array.shape[0]})", use_container_width=True)
            with col3:
                st.image(residual_map, caption="Difference / Residual Map", use_container_width=True)

            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            st.download_button(
                "⬇️ Download Restored Image",
                data=array_to_png_bytes(restored_array),
                file_name="automatically_restored.png",
                mime="image/png",
            )
        except Exception as error:
            st.error(f"Automatic denoising execution failed: {error}")
            st.exception(error)

else:
    # --- MANUAL WORKFLOW ---
    with st.sidebar:
        st.markdown("<h2>🎛️ Control Panel</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #94a3b8; font-size: 0.9rem;'>Configure synthetic degradation parameters for benchmark evaluation.</p>", unsafe_allow_html=True)
        st.markdown("---")
        noise_type = st.selectbox("📌 Noise Family", list(NOISE_LEVELS.keys()))
        severity = st.select_slider("⚡ Degradation Severity", options=["Low", "Medium", "High"], value="Medium")
        seed = st.number_input("🎲 Random Seed", min_value=0, max_value=999999, value=42, step=1)
        st.markdown("---")
        st.markdown("<p style='color: #64748b; font-size: 0.8rem;'>✨ Tip: Use high-resolution clean images to examine crisp structural preservation.</p>", unsafe_allow_html=True)

    st.markdown("### 📤 Upload Clean Reference Image")
    uploaded_file = st.file_uploader(
        "Choose a pristine reference image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        key="manual_upload",
    )

    if uploaded_file is not None:
        try:
            original_image = Image.open(uploaded_file).convert("RGB")
            clean_tensor = image_to_tensor(original_image)
            noisy_tensor = add_project_noise(clean_tensor, noise_type, severity, int(seed))

            basic_model = load_basic_model(noise_type, severity)
            dncnn_model = load_dncnn_model(noise_type)

            with st.spinner("Executing full-resolution comparative CNN inference..."):
                basic_output = denoise_full_resolution(noisy_tensor, basic_model)
                dncnn_output = denoise_full_resolution(noisy_tensor, dncnn_model)

            clean_array = tensor_to_array(clean_tensor)
            noisy_array = tensor_to_array(noisy_tensor)
            basic_array = tensor_to_array(basic_output)
            dncnn_array = tensor_to_array(dncnn_output)

            st.markdown("---")
            st.markdown(f"### 🧪 Benchmark Results — <span style='color: #2563eb;'>{noise_type} ({severity} Severity)</span>", unsafe_allow_html=True)
            
            noisy_filename = f"generated_{noise_type.lower()}_{severity.lower()}_noisy.png"
            col_dl1, col_dl2 = st.columns([1, 2])
            with col_dl1:
                st.download_button(
                    label=f"⬇️ Download Noisy Source",
                    data=array_to_png_bytes(noisy_array),
                    file_name=noisy_filename,
                    mime="image/png",
                )

            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            columns = st.columns(4)
            panels = [
                ("Pristine Reference", clean_array),
                ("Degraded Input", noisy_array),
                ("Basic CNN Output", basic_array),
                ("DnCNN Output", dncnn_array),
            ]
            for column, (label, array) in zip(columns, panels):
                with column:
                    st.image(array, caption=f"{label}\n({array.shape[1]}×{array.shape[0]})", use_container_width=True)

            basic_metrics = calculate_metrics(clean_array, basic_array)
            dncnn_metrics = calculate_metrics(clean_array, dncnn_array)
            noisy_metrics = calculate_metrics(clean_array, noisy_array)

            st.markdown("### 📈 Quantitative Quality Metrics")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Basic CNN PSNR", f"{basic_metrics[1]:.2f} dB")
            metric_columns[1].metric(
                "DnCNN PSNR",
                f"{dncnn_metrics[1]:.2f} dB",
                delta=f"{dncnn_metrics[1] - basic_metrics[1]:+.2f} dB vs Basic",
            )
            metric_columns[2].metric("Degraded Input PSNR", f"{noisy_metrics[1]:.2f} dB")

            table_data = {
                "Pipeline Stage": ["Degraded Input", "Basic CNN Baseline", "DnCNN Residual Model"],
                "MSE (↓)": [f"{noisy_metrics[0]:.5f}", f"{basic_metrics[0]:.5f}", f"{dncnn_metrics[0]:.5f}"],
                "PSNR dB (↑)": [f"{noisy_metrics[1]:.2f}", f"{basic_metrics[1]:.2f}", f"{dncnn_metrics[1]:.2f}"],
                "SSIM (↑)": [f"{noisy_metrics[2]:.4f}", f"{basic_metrics[2]:.4f}", f"{dncnn_metrics[2]:.4f}"],
            }
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            st.dataframe(table_data, hide_index=True, use_container_width=True)

            st.markdown("### 💾 Export Restored Assets")
            download_columns = st.columns(2)
            for column, label, array, filename in [
                (download_columns[0], "Download Basic CNN Output", basic_array, f"basic_cnn_{noise_type.lower()}_{severity.lower()}_denoised.png"),
                (download_columns[1], "Download DnCNN Output", dncnn_array, f"dncnn_{noise_type.lower()}_{severity.lower()}_denoised.png"),
            ]:
                with column:
                    st.download_button(
                        label,
                        data=array_to_png_bytes(array),
                        file_name=filename,
                        mime="image/png",
                    )
        except Exception as error:
            st.error(f"Manual benchmarking execution failed: {error}")
            st.exception(error)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>Intelligent Image Denoising Suite · Academic Research Demonstration</p>",
    unsafe_allow_html=True
)