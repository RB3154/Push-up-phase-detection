import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import tensorflow as tf
from src.model import build_model
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


def load_image(path, img_size=(224,224)):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image


def calculate_elbow_angle(image_path):
    """
    Uses MediaPipe to compute the angle at the right elbow.
    Returns angle in degrees or None if landmarks not detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if not results.pose_landmarks:
        return None
    lm = results.pose_landmarks.landmark
    # Coordinates in pixel space
    shoulder = np.array([
        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * img.shape[1],
        lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * img.shape[0]
    ])
    elbow = np.array([
        lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * img.shape[1],
        lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * img.shape[0]
    ])
    wrist = np.array([
        lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * img.shape[1],
        lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * img.shape[0]
    ])
    # Calculate angle
    v1 = shoulder - elbow
    v2 = wrist - elbow
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def predict_image(model, image_path, class_names=['up','down']):
    image = load_image(image_path)
    image = tf.expand_dims(image, 0)
    preds = model.predict(image)
    idx = tf.argmax(preds[0]).numpy()
    phase = class_names[idx]
    prob = float(preds[0][idx])
    return phase, prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to image file or directory')
    args = parser.parse_args()

    model = build_model(num_classes=2)
    model.load_weights('models/best_weights.weights.h5')

    def process_path(fpath):
        phase, prob = predict_image(model, fpath)
        angle = calculate_elbow_angle(fpath)
        # Pose-based hybrid override (Step 4)
        if angle is not None:
            # If model says 'up' but elbow angle is <100°, override to 'down'
            if phase == 'up' and angle < 100:
                phase = 'down'
            # If model says 'down' but elbow angle is >160°, override to 'up'
            elif phase == 'down' and angle > 160:
                phase = 'up'
        # Print result including angle
        # Safe formatting of angle
        if angle is None:
            angle_str = "N/A"
        else:
            angle_str = f"{angle:.1f}"
        print(f"{os.path.basename(fpath)}: {phase} ({prob:.2f}) | elbow_angle={angle_str}°")


    if os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            fpath = os.path.join(args.input, fname)
            if fpath.lower().endswith(('jpg','jpeg','png')):
                process_path(fpath)
    else:
        process_path(args.input)


if __name__ == '__main__':
    main()
