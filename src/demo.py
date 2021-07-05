# Importing Libraries & Packages

from  PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform
from keras.models import load_model
from skimage import metrics as skimage_metrics
from progressbar import progressbar
import random
import os
import cv2
import tempfile

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

def is_similar_frame(f1, f2, resize_to=(64, 64), thresh=0.5, return_score=False):
    thresh = float(os.getenv("FRAME_SIMILARITY_THRESH", thresh))
    
    if f1 is None or f2 is None:
        return False

    if isinstance(f1, str) and os.path.exists(f1):
        try:
            f1 = cv2.imread(f1)
        except Exception as ex:
            print(ex)
            return False

    if isinstance(f2, str) and os.path.exists(f2):
        try:
            f2 = cv2.imread(f2)
        except Exception as ex:
            print(ex)
            return False

    if resize_to:
        f1 = cv2.resize(f1, resize_to)
        f2 = cv2.resize(f2, resize_to)

    if len(f1.shape) == 3:
        f1 = f1[:, :, 0]

    if len(f2.shape) == 3:
        f2 = f2[:, :, 0]

    score = skimage_metrics.structural_similarity(f1, f2, multichannel=False)

    if return_score:
        return score

    if score >= thresh:
        return True

    return False

def get_interest_frames_from_video(video_path, frame_similarity_threshold=0.5,
                                   similarity_context_n_frames=3, skip_n_frames=0.5, 
                                   output_frames_to_dir=None,):
    skip_n_frames = float(skip_n_frames)
    important_frames = []
    fps = 0
    video_length = 0

    try:

        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_path.read())
        video = cv2.VideoCapture(tfile.name)
        #video = cv2.VideoCapture(video_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        if skip_n_frames < 1:
            skip_n_frames = int(skip_n_frames * fps)

        video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_i in range(length + 1):
            read_flag, current_frame = video.read()

            if not read_flag:
                break

            if skip_n_frames > 0:
                if frame_i % skip_n_frames != 0:
                    continue

            frame_i += 1

            found_similar = False
            for context_frame_i, context_frame in reversed(important_frames[-1 * similarity_context_n_frames :]):
                if is_similar_frame(context_frame, current_frame, thresh=frame_similarity_threshold):
                    found_similar = True
                    break

            if not found_similar:
                important_frames.append((frame_i, current_frame))
                if output_frames_to_dir:
                    if not os.path.exists(output_frames_to_dir):
                        os.mkdir(output_frames_to_dir)

                    output_frames_to_dir = output_frames_to_dir.rstrip("/")
                    cv2.imwrite(f"{output_frames_to_dir}/{str(frame_i).zfill(10)}.png",current_frame,)

    except Exception as ex:
        print(ex)

    return ([i[0] for i in important_frames], [i[1] for i in important_frames], fps, video_length,)

def process(x):
    frame = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    frame = frame/255.0
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    return frame

def sample_frames(x, y):
    sep = len(x) // 2
    f_half, s_half = x[:sep], x[sep:]
    f_frames = random.sample(f_half, round(len(f_half)*0.1))
    s_frames = random.sample(s_half, round(len(f_half)*0.2))
    frames = f_frames + s_frames
    return frames

def detect_video(model, video_path, mode="default", min_prob=0.6, batch_size=2, show_progress=True):
    frame_indices, frames, fps, video_length = get_interest_frames_from_video(video_path)
    
    if mode == "fast":
        frames = sample_frames(frames, frame_indices)
        #print(len(frames))
        frames = [process(frame) for frame in frames]
    else:
        frames = [process(frame) for frame in frames]

    all_results = {"metadata": {"fps": fps, "video_length": video_length, "video_path": video_path,}, "preds": {},}
    
    progress_func = progressbar
    
    if not show_progress:
        progress_func = dummy
        
    final_preds = []
    
    for _ in progress_func(range(int(len(frames) / batch_size) + 1)):
        batch = frames[:batch_size]
        batch_indices = frame_indices[:batch_size]
        frames = frames[batch_size:]
        frame_indices = frame_indices[batch_size:]

        if batch_indices:
            preds = [test_model.predict(np.expand_dims(frame, axis=0))[0] for frame in batch]
                                
            labelss = [config.LABELS[np.argmax(lab)] for lab in preds]
            scores = [str(np.max(sco)) for sco in preds]
                
            for frame_index, frame_score, frame_label in zip(frame_indices, scores, labelss):
                if frame_index not in all_results["preds"]:
                    all_results["preds"][frame_index] = []
                        
                if float(frame_score) < min_prob:
                    continue
                
                label = frame_label
                all_results["preds"][frame_index].append({"score": float(frame_score), "label": frame_label})
                      
                
    return all_results

def predict_video(vid_path, model, mode = 'default'):
    out = detect_video(test_model, vid_path, mode, min_prob=0.6, batch_size=2, show_progress=True)
    
    cnt = 0
    for i in out['preds'].values():
        if len(i) > 0:
            if i[0]['label'] == 'Porn' or i[0]['label'] == 'Hentai' :
                cnt+=1
    
    if cnt > 0:
        return("Video is Explicit!")
    else:
        return("Video is Safe!")

@st.cache
def predict(file_path):
    """
    Predict the Image to be Drawing/Hentai/Neutral/Porn/Sexy

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

file = st.file_uploader("Please upload a file", type = ["jpg", "jpeg", "png", "mp4"])
if file is not None:
    if file.type[:5] == "image":
        st.image(file, use_column_width = 'always', output_format = 'PNG')

        with st.spinner("Prediction in Progress..."):
            prediction = predict(file)
            st.success(prediction)

        if any([p in prediction.split(" ") for p in ["Hentai", "Porn", "Sexy"]]):
            image = Image.open(file)
            img_convert = np.array(image.convert('RGB'))
            slide = st.slider('Blur Quantity', 3, 81, 61, step=2)
            img_convert = cv2.cvtColor(img_convert, cv2.COLOR_RGB2BGR)
            blur_image = cv2.GaussianBlur(img_convert, (slide,slide), 0, 0)
            st.image(blur_image, channels='BGR', use_column_width = 'always') 
        else:
            pass
    else:
        st.video(file)
        with st.spinner("Prediction in Progress..."):
            prediction = predict_video(file, test_model)
            st.success(prediction)
