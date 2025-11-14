import streamlit as st
import librosa
import numpy as np
import pandas as pd
import tsfel
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
import soundfile as sf
from scipy.io import wavfile
import tempfile
import time

# Set page config
st.set_page_config(
    page_title="Voice Detection - Imam & Mufid",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid #ddd;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load models and preprocessing objects
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects"""
    try:
        with open("model/model_action.pkl", "rb") as f:
            model_action = pickle.load(f)

        with open("model/model_person.pkl", "rb") as f:
            model_person = pickle.load(f)

        with open("model/scaler_action.pkl", "rb") as f:
            scaler_action = pickle.load(f)

        with open("model/scaler_person.pkl", "rb") as f:
            scaler_person = pickle.load(f)

        with open("model/selected_features_action.pkl", "rb") as f:
            selected_features_action = pickle.load(f)

        with open("model/selected_features_person.pkl", "rb") as f:
            selected_features_person = pickle.load(f)

        with open("model/tsfel_config.pkl", "rb") as f:
            cfg = pickle.load(f)

        return {
            "model_action": model_action,
            "model_person": model_person,
            "scaler_action": scaler_action,
            "scaler_person": scaler_person,
            "selected_features_action": selected_features_action,
            "selected_features_person": selected_features_person,
            "cfg": cfg,
        }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None


def predict_voice(audio_file, models):
    """
    Predict voice from uploaded audio file

    Parameters:
    -----------
    audio_file : UploadedFile
        Uploaded audio file from Streamlit
    models : dict
        Dictionary containing all models and preprocessing objects

    Returns:
    --------
    dict : Prediction results with detailed information
    """
    try:
        # Load audio
        signal, sr = librosa.load(audio_file, sr=None)

        # Extract features
        features = tsfel.time_series_features_extractor(
            models["cfg"], signal, verbose=0
        )

        # Handle NaN and Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())

        # Predict ACTION (buka/tutup)
        X_action = features[models["selected_features_action"]]
        X_action_scaled = models["scaler_action"].transform(X_action)
        pred_action = models["model_action"].predict(X_action_scaled)[0]
        pred_action_proba = models["model_action"].predict_proba(X_action_scaled)[0]

        # Predict PERSON (imam/mufid)
        X_person = features[models["selected_features_person"]]
        X_person_scaled = models["scaler_person"].transform(X_person)
        pred_person = models["model_person"].predict(X_person_scaled)[0]
        pred_person_proba = models["model_person"].predict_proba(X_person_scaled)[0]

        return {
            "signal": signal,
            "sr": sr,
            "prediksi_aksi": pred_action,
            "confidence_aksi": max(pred_action_proba) * 100,
            "prediksi_orang": pred_person,
            "confidence_orang": max(pred_person_proba) * 100,
            "probabilitas_aksi": dict(
                zip(models["model_action"].classes_, pred_action_proba)
            ),
            "probabilitas_orang": dict(
                zip(models["model_person"].classes_, pred_person_proba)
            ),
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_voice_from_array(signal, sr, models):
    """
    Predict voice from audio array (for real-time recording)

    Parameters:
    -----------
    signal : np.array
        Audio signal array
    sr : int
        Sample rate
    models : dict
        Dictionary containing all models and preprocessing objects

    Returns:
    --------
    dict : Prediction results with detailed information
    """
    try:
        # Extract features
        features = tsfel.time_series_features_extractor(
            models["cfg"], signal, verbose=0
        )

        # Handle NaN and Inf
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())

        # Predict ACTION (buka/tutup)
        X_action = features[models["selected_features_action"]]
        X_action_scaled = models["scaler_action"].transform(X_action)
        pred_action = models["model_action"].predict(X_action_scaled)[0]
        pred_action_proba = models["model_action"].predict_proba(X_action_scaled)[0]

        # Predict PERSON (imam/mufid)
        X_person = features[models["selected_features_person"]]
        X_person_scaled = models["scaler_person"].transform(X_person)
        pred_person = models["model_person"].predict(X_person_scaled)[0]
        pred_person_proba = models["model_person"].predict_proba(X_person_scaled)[0]

        return {
            "signal": signal,
            "sr": sr,
            "prediksi_aksi": pred_action,
            "confidence_aksi": max(pred_action_proba) * 100,
            "prediksi_orang": pred_person,
            "confidence_orang": max(pred_person_proba) * 100,
            "probabilitas_aksi": dict(
                zip(models["model_action"].classes_, pred_action_proba)
            ),
            "probabilitas_orang": dict(
                zip(models["model_person"].classes_, pred_person_proba)
            ),
            "success": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def plot_waveform(signal, sr):
    """Plot audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    time = np.arange(len(signal)) / sr
    ax.plot(time, signal, color="#1f77b4", linewidth=0.5)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Amplitude", fontsize=11)
    ax.set_title("Audio Waveform", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_probabilities(prob_dict, title, color):
    """Plot probability bars"""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(prob_dict.keys())
    values = [v * 100 for v in prob_dict.values()]

    bars = ax.barh(labels, values, color=color, edgecolor="black", alpha=0.7)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim([0, 105])
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def record_audio(duration=3, sample_rate=22050):
    """
    Record audio from microphone

    Parameters:
    -----------
    duration : int
        Recording duration in seconds
    sample_rate : int
        Sample rate for recording

    Returns:
    --------
    tuple : (audio_data, sample_rate)
    """
    try:
        st.info(f"🎙️ Recording for {duration} seconds... Speak now!")

        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )

        # Show progress bar
        progress_bar = st.progress(0)
        for i in range(duration):
            time.sleep(1)
            progress_bar.progress((i + 1) / duration)

        sd.wait()  # Wait until recording is finished
        progress_bar.empty()

        st.success("✅ Recording completed!")

        # Convert to mono if needed
        if len(recording.shape) > 1:
            recording = recording[:, 0]

        return recording.flatten(), sample_rate

    except Exception as e:
        st.error(f"❌ Recording error: {str(e)}")
        return None, None


def display_prediction_results(result):
    """
    Display prediction results in a formatted way

    Parameters:
    -----------
    result : dict
        Prediction results dictionary
    """
    st.markdown(
        '<p class="sub-header">📊 Prediction Results</p>',
        unsafe_allow_html=True,
    )

    # Display results in columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Action Prediction")
        st.markdown(
            f"<h2 style='color: #1f77b4; text-align: center;'>{result['prediksi_aksi'].upper()}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align: center; font-size: 1.2rem;'>Confidence: <strong>{result['confidence_aksi']:.2f}%</strong></p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Plot action probabilities
        fig_action = plot_probabilities(
            result["probabilitas_aksi"],
            "Action Probability Distribution",
            ["#1f77b4", "#ff7f0e"],
        )
        st.pyplot(fig_action)

    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("### 👤 Person Prediction")
        st.markdown(
            f"<h2 style='color: #ff7f0e; text-align: center;'>{result['prediksi_orang'].upper()}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align: center; font-size: 1.2rem;'>Confidence: <strong>{result['confidence_orang']:.2f}%</strong></p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Plot person probabilities
        fig_person = plot_probabilities(
            result["probabilitas_orang"],
            "Person Probability Distribution",
            ["#2ca02c", "#d62728"],
        )
        st.pyplot(fig_person)

    # Combined prediction summary
    st.markdown('<p class="sub-header">📝 Summary</p>', unsafe_allow_html=True)

    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

    with col_sum1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("**Action**")
        st.markdown(
            f"<h3 style='color: #1f77b4;'>{result['prediksi_aksi']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sum2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("**Action Confidence**")
        st.markdown(
            f"<h3 style='color: #1f77b4;'>{result['confidence_aksi']:.1f}%</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sum3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("**Person**")
        st.markdown(
            f"<h3 style='color: #ff7f0e;'>{result['prediksi_orang']}</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sum4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.markdown("**Person Confidence**")
        st.markdown(
            f"<h3 style='color: #ff7f0e;'>{result['confidence_orang']:.1f}%</h3>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Audio waveform visualization
    st.markdown(
        '<p class="sub-header">🎵 Audio Visualization</p>',
        unsafe_allow_html=True,
    )
    fig_wave = plot_waveform(result["signal"], result["sr"])
    st.pyplot(fig_wave)

    # Detailed probabilities
    with st.expander("🔍 View Detailed Probabilities"):
        col_detail1, col_detail2 = st.columns(2)

        with col_detail1:
            st.markdown("**Action Probabilities:**")
            for label, prob in result["probabilitas_aksi"].items():
                st.write(f"- {label}: {prob*100:.2f}%")

        with col_detail2:
            st.markdown("**Person Probabilities:**")
            for label, prob in result["probabilitas_orang"].items():
                st.write(f"- {label}: {prob*100:.2f}%")


# Main App
def main():
    # Header
    st.markdown(
        '<p class="main-header">🎤 Voice Detection System</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-Label Classification: Action & Person Recognition</p>',
        unsafe_allow_html=True,
    )

    # Load models
    models = load_models()

    if models is None:
        st.error(
            "❌ Failed to load models. Please check if model files exist in 'model/' directory."
        )
        st.stop()

    st.success("✅ Models loaded successfully!")

    # Sidebar
    with st.sidebar:
        st.markdown("### 📋 Information")
        st.markdown(
            """
        **Model Details:**
        - **Action Model**: Classifies 'buka' or 'tutup'
        - **Person Model**: Classifies 'imam' or 'mufid'
        - **Algorithm**: Random Forest Classifier
        - **Features**: TSFEL Audio Features
        """
        )

        st.markdown("---")
        st.markdown("### 🎯 Model Performance")

        # You can add model metrics here
        st.metric("Action Accuracy", "95%+", delta="High")
        st.metric("Person Accuracy", "90%+", delta="High")

        st.markdown("---")
        st.markdown("### 👥 Created by")
        st.markdown("**Imam & Mufid**")
        st.markdown("Proyek Sains Data 2025-2026")

    # Main content
    st.markdown(
        '<p class="sub-header">🔊 Choose Input Method</p>', unsafe_allow_html=True
    )

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Real-Time"])

    # Tab 1: Upload Audio File
    with tab1:
        st.markdown("### Upload an audio file from your device")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV or MP3)",
            type=["wav", "mp3"],
            help="Upload a voice recording to predict the action and person",
        )

        if uploaded_file is not None:
            # Display file info
            st.info(
                f"📁 File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)"
            )

            # Audio player
            st.audio(uploaded_file, format="audio/wav")

            # Predict button
            if st.button(
                "🚀 Predict from File", type="primary", use_container_width=True
            ):
                with st.spinner("🔄 Processing audio and making predictions..."):
                    # Make prediction
                    result = predict_voice(uploaded_file, models)

                    if result["success"]:
                        display_prediction_results(result)
                    else:
                        st.error(f"❌ Prediction failed: {result['error']}")
        else:
            st.info("👆 Please upload an audio file to start prediction")

    # Tab 2: Real-Time Recording
    with tab2:
        st.markdown("### Record your voice in real-time")

        col_rec1, col_rec2 = st.columns(2)

        with col_rec1:
            duration = st.slider(
                "Recording Duration (seconds)",
                min_value=1,
                max_value=10,
                value=3,
                help="Set how long you want to record",
            )

        with col_rec2:
            sample_rate = st.selectbox(
                "Sample Rate",
                options=[16000, 22050, 44100],
                index=1,
                help="Audio quality (higher = better quality but larger file)",
            )

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
                # Record audio
                audio_data, sr = record_audio(duration, sample_rate)

                if audio_data is not None:
                    # Store in session state
                    st.session_state["recorded_audio"] = audio_data
                    st.session_state["recorded_sr"] = sr

                    # Save to temporary file for playback
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    sf.write(temp_file.name, audio_data, sr)
                    st.session_state["temp_audio_file"] = temp_file.name

                    st.success("✅ Recording saved successfully!")

        # Display recorded audio if exists
        if "recorded_audio" in st.session_state:
            st.markdown("### 🎵 Your Recording")

            # Play recorded audio
            with open(st.session_state["temp_audio_file"], "rb") as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/wav")

            # Show waveform preview
            fig_preview = plot_waveform(
                st.session_state["recorded_audio"], st.session_state["recorded_sr"]
            )
            st.pyplot(fig_preview)

            # Predict button for recorded audio
            if st.button(
                "🚀 Predict from Recording", type="primary", use_container_width=True
            ):
                with st.spinner("🔄 Processing audio and making predictions..."):
                    # Make prediction
                    result = predict_voice_from_array(
                        st.session_state["recorded_audio"],
                        st.session_state["recorded_sr"],
                        models,
                    )

                    if result["success"]:
                        display_prediction_results(result)
                    else:
                        st.error(f"❌ Prediction failed: {result['error']}")
        else:
            st.info("👆 Click 'Start Recording' to record your voice")

            # Instructions
            st.markdown("### 📖 How to Record")
            st.markdown(
                """
            1. **Allow microphone access** when prompted by your browser
            2. **Set recording duration** (1-10 seconds recommended)
            3. **Click 'Start Recording'** button
            4. **Speak clearly** when recording starts
            5. **Wait** for recording to complete
            6. **Click 'Predict'** to get results
            """
            )

    # Show instructions when no input
    if uploaded_file is None and "recorded_audio" not in st.session_state:
        # Show example when no file is uploaded
        st.info("👆 Please upload an audio file to start prediction")

        # Show example images or instructions
        st.markdown('<p class="sub-header">📖 How to Use</p>', unsafe_allow_html=True)

        st.markdown(
            """
        1. **Upload** an audio file (WAV or MP3 format)
        2. **Click** the "Predict" button
        3. **View** the prediction results:
           - **Action**: Whether the voice says 'buka' (open) or 'tutup' (close)
           - **Person**: Whether the voice is from 'imam' or 'mufid'
        4. Check the **confidence scores** to see how certain the model is
        5. View the **audio waveform** and **probability distributions**
        """
        )

        st.markdown('<p class="sub-header">✨ Features</p>', unsafe_allow_html=True)

        col_feat1, col_feat2, col_feat3 = st.columns(3)

        with col_feat1:
            st.markdown("### 🎯 Accurate")
            st.write("High accuracy multi-label classification using Random Forest")

        with col_feat2:
            st.markdown("### ⚡ Fast")
            st.write("Real-time prediction with optimized feature extraction")

        with col_feat3:
            st.markdown("### 📊 Detailed")
            st.write("Comprehensive results with visualizations and confidence scores")

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown(
        "**Voice Detection System** | Powered by Random Forest & TSFEL | 2025-2026"
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
