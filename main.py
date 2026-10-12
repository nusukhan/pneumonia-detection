from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# website ko backend se baat karne ki ijazat
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# model file load karo (ek hi baar, shuru mein)
model = tf.keras.models.load_model('pneumonia_model.keras')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. upload hui image parho
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')

    # 2. model ke layak banao (wahi 150x150, yaad hai?)
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 3. model se prediction lo
    prediction = model.predict(img_array)[0][0]

    # 4. jawab taiyaar karo
    if prediction > 0.5:
        label = "Pneumonia"
        confidence = float(prediction)
    else:
        label = "Normal"
        confidence = float(1 - prediction)

    return {"prediction": label, "confidence": round(confidence * 100, 2)}