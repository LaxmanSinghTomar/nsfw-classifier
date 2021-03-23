# Impprting Libraries & Packages

import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, 'src/')

import config
import argparse

vid_model = load_model(config.MODEL_PATH)

labels = {0 : "Drawing", 1 : "Hentai", 2 : "Neutral", 3: "Porn", 4: "Sexy"}
size = 128
Q = deque(maxlen=size)

def predict(file_path):
    vs = cv2.VideoCapture(file_path)
    writer = None
    (W, H) = (None, None)
 
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
 
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
 
        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
    
        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame/255.0
        frame = cv2.resize(frame, (224, 224)).astype("float32")
    
        #frame -= mean
    
        # make predictions on the frame and then update the predictions
        # queue
        preds = vid_model.predict(np.expand_dims(frame, axis=0))[0]
        print(preds)
        Q.append(preds)

        # perform prediction averaging over the current history of
        # previous predictions

        results = np.array(Q).mean(axis=0)
        i = np.argmax(preds)
        label = labels[i]
        # draw the activity on the output frame
        text = "activity: {}:".format(label)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
        
        output_vid = "output/" + file_path.split('/')[-1][:-4] + "--output.mp4"

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            writer = cv2.VideoWriter(output_vid, fourcc, 30, (W, H), True)

        # write the output frame to disk
        writer.write(output)

        # show the output image
        #cv2.imshow("Output", output)
        #cv2_imshow(output)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        
    # release the file pointers
    print("[INFO] cleaning up...")
    # writer.release()
    vs.release()


if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # currently, we only need filepath to the image
    parser.add_argument(
        "--file_path",
        type = str
    )

    # read the arguments from the command line
    args = parser.parse_args()

    # run the predict specified by command line arguments
    predict(
        file_path=args.file_path
    )