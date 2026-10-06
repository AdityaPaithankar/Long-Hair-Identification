import os
import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Long Hair Identification",
    page_icon="💇",
    layout="wide"
)


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = (160, 160)
THRESHOLD = 0.70

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "hair_model_best.keras"

CASCADE_PATH = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        margin-top: 20px;
    }

    .long-box {
        background: #e8f8ef;
        border: 2px solid #28a745;
    }

    .short-box {
        background: #fff3df;
        border: 2px solid #f39c12;
    }

    .confidence {
        font-size: 38px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.code(str(e))

    st.stop()


# =========================================================
# FACE DETECTOR
# =========================================================

@st.cache_resource
def load_face_detector():

    detector = cv2.CascadeClassifier(
        CASCADE_PATH
    )

    if detector.empty():

        raise RuntimeError(
            "Face detector could not be loaded."
        )

    return detector


face_detector = load_face_detector()


# =========================================================
# PREDICT HAIR
# =========================================================

def predict_hair(image):

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        IMG_SIZE
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    score = float(
        model.predict(
            image,
            verbose=0
        )[0][0]
    )

    # Class 0 = Long
    # Class 1 = Short
    #
    # Sigmoid score = probability of Short

    if score >= THRESHOLD:

        return "SHORT", score

    return "LONG", 1 - score


# =========================================================
# MULTIPLE FACE + HAIR PREDICTION
# =========================================================

def process_frame(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    regions = []
    boxes = []

    height, width = frame.shape[:2]

    for (x, y, w, h) in faces:

        # Expand face area to capture hair
        expand_x = int(w * 0.45)
        expand_top = int(h * 0.75)
        expand_bottom = int(h * 1.8)

        x1 = max(
            0,
            x - expand_x
        )

        y1 = max(
            0,
            y - expand_top
        )

        x2 = min(
            width,
            x + w + expand_x
        )

        y2 = min(
            height,
            y + h + expand_bottom
        )

        roi = frame[
            y1:y2,
            x1:x2
        ]

        if roi.size == 0:
            continue

        regions.append(roi)

        boxes.append(
            (x1, y1, x2, y2)
        )


    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    predictions = []

    for roi in regions:

        label, confidence = predict_hair(
            roi
        )

        predictions.append(
            (label, confidence)
        )


    # -----------------------------------------------------
    # DRAW
    # -----------------------------------------------------

    for i, box in enumerate(boxes):

        if i >= len(predictions):
            continue

        x1, y1, x2, y2 = box

        label, confidence = predictions[i]

        confidence_percent = (
            confidence * 100
        )

        if label == "LONG":

            color = (0, 220, 120)

        else:

            color = (255, 170, 50)


        # Face/hair box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )


        # Label

        text = (
            f"Person {i + 1}: "
            f"{label} "
            f"{confidence_percent:.1f}%"
        )

        label_y = max(
            30,
            y1 - 10
        )

        cv2.rectangle(
            frame,
            (x1, label_y - 30),
            (x1 + 245, label_y + 5),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            text,
            (x1 + 5, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )


    # -----------------------------------------------------
    # PERSON COUNT
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (190, 55),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        f"People: {len(boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


# =========================================================
# WEBRTC CALLBACK
# =========================================================

def video_frame_callback(frame):

    img = frame.to_ndarray(
        format="bgr24"
    )

    processed = process_frame(
        img
    )

    return av.VideoFrame.from_ndarray(
        processed,
        format="bgr24"
    )


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">'
    '💇 Long Hair Identification'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Long vs Short Hair Classification'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs(
    [
        "📤 Image Upload",
        "📷 Live Webcam"
    ]
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

with tab1:

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        image_cv = np.array(
            image
        )

        image_cv = cv2.cvtColor(
            image_cv,
            cv2.COLOR_RGB2BGR
        )

        label, confidence = predict_hair(
            image_cv
        )

        confidence_percent = (
            confidence * 100
        )


        if label == "LONG":

            st.markdown(
                f'<div class="result-box long-box">'
                f'<h2>💇 Long Hair</h2>'
                f'<div class="confidence">'
                f'{confidence_percent:.2f}%'
                f'</div>'
                f'<p>Model Confidence</p>'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="result-box short-box">'
                f'<h2>✂️ Short Hair</h2>'
                f'<div class="confidence">'
                f'{confidence_percent:.2f}%'
                f'</div>'
                f'<p>Model Confidence</p>'
                f'</div>',
                unsafe_allow_html=True
            )


        if confidence_percent < 60:

            st.warning(
                "⚠️ Low confidence. "
                "Try a clearer image."
            )

        elif confidence_percent < 80:

            st.info(
                "ℹ️ Moderate confidence."
            )

        else:

            st.success(
                "✅ High confidence prediction."
            )

        st.progress(
            min(
                int(confidence_percent),
                100
            )
        )


# =========================================================
# LIVE WEBCAM
# =========================================================

with tab2:

    st.subheader(
        "📷 Live Multi-Person Detection"
    )

    st.write(
        "Click START and allow browser camera permission."
    )

    webrtc_streamer(
        key="hair-detection-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        },
        async_processing=True
    )


# =========================================================
# MODEL INFO
# =========================================================

with st.expander("🔍 Model Information"):

    st.write(
        "**Model:** MobileNetV2"
    )

    st.write(
        "**Input:** 160 × 160"
    )

    st.write(
        "**Classes:** Long / Short"
    )

    st.write(
        "**Threshold:** 0.70"
    )

    st.write(
        "**Model file:** hair_model_best.keras"
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "Long Hair Identification • "
    "MobileNetV2 • Multi-Person Computer Vision"
)