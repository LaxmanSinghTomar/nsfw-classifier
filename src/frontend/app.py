# Importing Required Libraries & Packages

import requests
import streamlit as st


st.image("NSFW Classifier.png")
st.write("""
# NSFW Classifier""")
st.write("This is a simple image classification web app to predict whether image is NSFW or not!")

file = st.file_uploader("Please upload an image file", type=["jpg", "jpeg", "png"])
if file is not None:
    st.image(file, use_column_width = 'always', output_format = 'PNG')
    files = {"file": file.getvalue()}
    with st.spinner("Prediction in Progress"):
        res = requests.post(f"http://0.0.0.0:8000/predict/image", files = files)
        resp = res.text
        st.success(resp)