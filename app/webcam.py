import cv2
import tensorflow as tf
import numpy as np
import os
import time


# =========================================================
# SETTINGS
# =========================================================

IMG_SIZE = (160, 160)
THRESHOLD = 0.70
PREDICT_EVERY = 5

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "model", "hair_model_best.keras"
)

CASCADE_PATH = cv2.data.haarcascades + \
    "haarcascade_frontalface_default.xml"


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded")


# =========================================================
# FACE DETECTOR
# =========================================================

face_detector = cv2.CascadeClassifier(CASCADE_PATH)

if face_detector.empty():
    print("❌ Face detector failed")
    exit()

print("✅ Face detector loaded")


# =========================================================
# PREDICTION
# =========================================================

def predict_hair_batch(regions):

    images = []

    for region in regions:

        region = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2RGB
        )

        region = cv2.resize(
            region,
            IMG_SIZE
        )

        images.append(region)

    if not images:
        return []

    batch = np.array(images)

    scores = model.predict(
        batch,
        verbose=0
    ).flatten()

    results = []

    for score in scores:

        if score >= THRESHOLD:
            label = "SHORT"
            confidence = score
        else:
            label = "LONG"
            confidence = 1 - score

        results.append(
            (label, confidence)
        )

    return results


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Camera could not be opened")
    exit()

print("✅ Camera connected")
print("Press Q to quit")


# =========================================================
# VARIABLES
# =========================================================

frame_count = 0

results = []

previous_time = time.time()
fps = 0


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Frame read failed")
        break

    frame_count += 1

    # Mirror camera
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # FACE DETECTION
    # =====================================================

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    # =====================================================
    # CREATE HAIR REGIONS
    # =====================================================

    regions = []
    boxes = []

    for (x, y, w, h) in faces:

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
            frame.shape[1],
            x + w + expand_x
        )

        y2 = min(
            frame.shape[0],
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


    # =====================================================
    # PREDICT EVERY N FRAMES
    # =====================================================

    if frame_count % PREDICT_EVERY == 0:

        results = predict_hair_batch(
            regions
        )


    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:

        fps = 1 / elapsed

    previous_time = current_time


    # =====================================================
    # DRAW RESULTS
    # =====================================================

    for i, box in enumerate(boxes):

        if i >= len(results):
            continue

        x1, y1, x2, y2 = box

        label, confidence = results[i]

        confidence_percent = confidence * 100


        # -----------------------------------------------
        # COLOR
        # -----------------------------------------------

        if label == "LONG":

            color = (0, 220, 120)

        else:

            color = (255, 170, 50)


        # -----------------------------------------------
        # PERSON LABEL
        # -----------------------------------------------

        person_text = (
            f"Person {i + 1}: "
            f"{label} "
            f"{confidence_percent:.1f}%"
        )


        # -----------------------------------------------
        # BOX
        # -----------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # -----------------------------------------------
        # LABEL BACKGROUND
        # -----------------------------------------------

        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 32)),
            (x1 + 230, y1),
            (20, 20, 20),
            -1
        )


        # -----------------------------------------------
        # LABEL TEXT
        # -----------------------------------------------

        cv2.putText(
            frame,
            person_text,
            (x1 + 5, y1 - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )


    # =====================================================
    # HEADER
    # =====================================================

    cv2.rectangle(
        frame,
        (0, 0),
        (300, 65),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        f"People: {len(boxes)}",
        (15, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        2
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    cv2.imshow(
        "Long Hair Identification - Multi Person",
        frame
    )


    # =====================================================
    # EXIT
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()

print("✅ Camera stopped")