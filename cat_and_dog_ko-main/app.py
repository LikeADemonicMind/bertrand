from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
import os
from io import BytesIO
from PIL import Image
import base64

import keras
import mlflow

app = Flask(__name__)

# Configuration de MLflow
mlflow.set_tracking_uri("http://localhost:5000")

MODEL_PATH = os.path.join("models", "model.keras")

model = load_model(MODEL_PATH)
model.make_predict_function()

def model_predict(img, model):
    img_resized = img.resize((128, 128))  
    x = keras.utils.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Prédiction du modèle
    preds = model.predict(x)
    
    # Enregistrer les prédictions et les métriques avec MLflow
    with mlflow.start_run():
        mlflow.log_metric("Chien_probability", preds[0][1])
        mlflow.log_metric("Chat_probability", preds[0][0])
        mlflow.log_param("image_size", "128x128")
        
        result = "Chien" if preds[0][0] < 0.5 else "Chat"
        mlflow.log_param("result", result)
        
    return preds

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        buffered_img = BytesIO(f.read())
        img = Image.open(buffered_img)

        base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

        preds = model_predict(img, model)
        result = "Chien" if preds[0][0] < 0.5 else "Chat"
        
        return render_template('result.html', result=result, image_base64_front=base64_img)
    
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
