import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from feature_extraction import extract_features

# โหลดโมเดล
model = load_model("best_model-4.keras")

st.title("🩺 Heart Murmur Detection")
st.write("อัปโหลดไฟล์เสียงหัวใจ (.wav) เพื่อให้โมเดลตรวจวินิจฉัย")

uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # โหลดเสียง
    y, sr = librosa.load(uploaded_file, sr=None)

    # แปลงเสียงเป็น features
    features = extract_features(y, sr)
    features = np.expand_dims(features, axis=0)

    # ทำนายผล
    prediction = model.predict(features)[0][0]
    result = "🩺 Murmur Detected" if prediction >= 0.5 else "✅ Normal Heart Sound"
    st.markdown(f"### ผลลัพธ์: {result}")
    st.write(f"Confidence: `{prediction:.2f}`")
