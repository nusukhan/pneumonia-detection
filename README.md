🫁 Pneumonia Detection from Chest X-Rays

An AI-powered web application that uses a Convolutional Neural Network (CNN) to analyze chest X-ray images and classify them as Normal or Pneumonia, along with a confidence score.

⚠️ For educational purposes only. This is not a medical diagnostic tool. Always consult a qualified healthcare professional.

📌 Overview

This project takes a chest X-ray image, runs it through a trained CNN, and returns a prediction with a confidence percentage — all through a clean web interface.

Final flow:

Upload X-Ray  →  CNN Model  →  Analysis  →  Normal / Pneumonia + Confidence Score
🧠 How It Works
Upload — User uploads a chest X-ray through the web page (drag & drop or file picker).
Backend (FastAPI) — The image is sent to a FastAPI server that acts as the bridge between the website and the model.
CNN Model — The trained model analyzes the image and outputs a probability.
Result — The prediction (Normal/Pneumonia) and confidence score are displayed, with a color-coded confidence bar.
🛠️ Tech Stack
Layer	Tools
Model	TensorFlow / Keras (CNN)
Training	Google Colab (GPU)
Dataset	Chest X-Ray Pneumonia (Kaggle)
Backend	FastAPI + Uvicorn
Frontend	HTML, CSS, JavaScript
Image processing	Pillow, NumPy
📊 Dataset & The Imbalance Problem

The dataset contains 5,216 training images, but it is imbalanced:

Class	Images	Ratio
Normal	1,341	1
Pneumonia	3,875	~2.9

This imbalance matters: a lazy model that predicts "Pneumonia" for everything would still score ~74% accuracy — which is why accuracy alone is misleading.

Solution: Instead of deleting data, I used class weights to tell the model to pay more attention to the under-represented Normal class.

⚠️ Note on the validation set: The dataset's built-in val folder has only 16 images, which is too small to be reliable. I created a proper validation set by splitting 20% off the training data (with a fixed seed to prevent overlap between train and validation).

🏗️ Model Architecture

A Sequential CNN built from scratch:

Rescaling (1/255)          → normalize pixels to 0–1
Data Augmentation          → random rotation + zoom (NO flip — see note below)
Conv2D (32) + MaxPooling
Conv2D (64) + MaxPooling
Conv2D (128) + MaxPooling
Flatten
Dense (128) + Dropout(0.5) → reduce overfitting
Dense (1, sigmoid)         → output probability (0 = Normal, 1 = Pneumonia)

Total parameters: ~4.8 million

🩺 Why no horizontal flip? The heart is on the left side of the chest. Flipping an X-ray left-right would create anatomically impossible images that never occur in real life, so it would teach the model wrong patterns.

📈 Results

Trained for 10 epochs with class weights. Evaluated on the unseen test set (624 images):

Metric	Score
Test Accuracy	91%
Pneumonia Recall	0.96
Normal Recall	0.82
Confusion Matrix
	Predicted Normal	Predicted Pneumonia
Actual Normal	193 ✅	41 ❌
Actual Pneumonia	17 ❌	373 ✅
Why these numbers matter
Pneumonia recall (0.96) means the model catches 96 out of every 100 real pneumonia cases. This is the most important metric here — missing a sick patient (a False Negative, the 17 cases above) is the most dangerous mistake in medical AI.
Accuracy (91%) hides weaknesses. The confusion matrix reveals the model is weaker at identifying Normal cases (82% recall) — something accuracy alone would never show. This is exactly why precision and recall matter more than accuracy on imbalanced data.
🚀 How to Run Locally
1. Clone the repo
bash
git clone https://github.com/nusukhan/pneumonia-detection.git
cd pneumonia-detection
2. Install dependencies
bash
pip install fastapi uvicorn tensorflow pillow python-multipart
3. Start the backend
bash
uvicorn main:app --reload

The server runs at http://127.0.0.1:8000.

4. Open the frontend

Open index.html in your browser, upload a chest X-ray, and click Analyze X-Ray.

📁 Project Structure
pneumonia-detection/
├── main.py                  # FastAPI backend (loads model, /predict endpoint)
├── index.html               # Web frontend (upload + results UI)
├── pneumonia_model.keras    # Trained CNN model
├── train_model.ipynb        # Colab notebook (data prep, training, evaluation)
└── README.md
🔮 Future Improvements
Transfer learning (e.g. VGG16, ResNet) to improve accuracy beyond 91%
More diverse data so the model generalizes to X-rays from different machines
Deployment to a cloud platform (Hugging Face Spaces / Render) for public access
👩‍💻 Author

Nusrat Anwar — GitHub · Portfolio