# Importing Libraries & Packages

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform
from keras.models import load_model

import argparse
import sys

import streamlit as st
import config

test_model = load_model(config.MODEL_PATH)


def load(filename):
    """
    Load Images using the provided 'filename'.

    Note:
        Input filename must be a string path.

    Args:
        filename (str): Path to the filename.

    Returns:
        An array representation of Image.
    """
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (224, 224, 3))
    np_image = np.expand_dims(np_image, axis=0)
    img = mpimg.imread(filename)
    return np_image


def predict(file_path):
    """
    Predict the Image to be Drawing/Hentai/Neutral/Pornn/Sexy

    Note:
        Input Path must be string.

    Args:
        file_path(str) : Path to the filename.

    Returns:
        Predicted Label and a numpy array containing the confidence scores the model has in making a prediction.
    """
    image = load(file_path)
    ans = test_model.predict(image)
    mapping = {0: "Drawing", 1: "Hentai", 2: "Neutral", 3: "Porn", 4: "Sexy"}
    new_ans = np.argmax(ans[0])
    pred = "{} with {} Probability!".format(mapping[new_ans], ans[0][new_ans]) 
    return pred

st.image("NSFW Classifier.png")
st.write("""
# NSFW Classifier""")
st.write("This is a simple image classification web app to predict whether image is NSFW or not!")

file = st.file_uploader("Please upload an image file", type=["jpg", "jpeg", "png"])
if file is not None:
    st.image(file, use_column_width = 'always', output_format = 'PNG')
    with st.spinner("Prediction in Progress..."):
        prediction = predict(file)
        st.success(prediction)
    
    
