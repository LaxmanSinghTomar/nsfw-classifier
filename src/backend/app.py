# Importing Libraries and Packages
import uvicorn
from fastapi import FastAPI, File, UploadFile

from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform
from keras.models import load_model


import argparse
import sys

sys.path.insert(1, "src/")
import config

app = FastAPI()

test_model = load_model(config.MODEL_PATH)

def load(file):
    """
    Load Images using the provided file.

    Note:
        Input file must be a BytesIO Object.

    Args:
        filename (str): BytesIO Object.

    Returns:
        An array representation of Image.
    """
    np_image = Image.open(BytesIO(file))
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def predict(file_path):
    """
    Predict the Image to be Drawing/Hentai/Neutral/Pornn/Sexy

    Note:
        Input must be an Numpy Array Image.

    Args:
        file_path(np.array) : Numpy Array Image.

    Returns:
        Predicted Label and the confidence score the model has in making a prediction.
    """
    ans = test_model.predict(file_path)
    mapping = {0: "Drawing", 1: "Hentai", 2: "Neutral", 3: "Porn", 4: "Sexy"}
    new_ans = np.argmax(ans[0])
    pred = "{} with {} Probability!".format(mapping[new_ans], ans[0][new_ans]) 
    return pred

@app.post("/predict/image")
async def predict_api(file:UploadFile = File(...)):
    img = load(await file.read())
    prediction = predict(img)
    return prediction

if __name__ == '__main__':
    uvicorn.run(app, port = 8000, host = "0.0.0.0", debug=True)