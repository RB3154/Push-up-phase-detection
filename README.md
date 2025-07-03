# 🏋️‍♂️ Push-up Phase Detection AI

**Detect. Analyze. Improve.**  
This project uses **deep learning** and **computer vision** to classify the phases of a push-up – identifying whether the person is in the "up" or "down" position. Built using Python, TensorFlow, and MobileNetV2, this AI is here to make your workouts smarter 💪

---

## 🚀 Features
- 🎯 Classifies push-up phases (`UP` ⬆️ or `DOWN` ⬇️)
- 🧠 Uses **MobileNetV2** for efficient model performance
- 🔢 Calculates **elbow angles** using keypoint data
- 📊 Visualizes predictions with **Matplotlib**
- 💾 Includes **pre-trained model** – no training required!

---

## 🧠 Tech Stack
- **Language:** Python 🐍
- **Libraries:** TensorFlow, NumPy, Pandas, Matplotlib, open-cv, mediapipe
- **Model:** MobileNetV2 (pre-trained on ImageNet, fine-tuned for push-up classification)

---
## 🏃‍♂️ Run Predictions
- Use the following command to run the prediction script on your input image file:
- python src/predict.py --input path_to_your_image
📝 Replace path_to_your_image_or_video with the actual path to your image file.
