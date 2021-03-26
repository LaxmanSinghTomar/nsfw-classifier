# Importing Libraries & Packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import sys

sys.path.insert(1, "src/")
import config


num_of_test_samples = 10000
batch_size = 64
size = config.SIZE

validation_data_generation = ImageDataGenerator(rescale=1.0 / 255)  # need float values

test_generator = validation_data_generation.flow_from_directory(
    config.TEST_PATH,
    target_size=(size, size),
    class_mode="categorical",
    batch_size=64,
    shuffle=False,
)

model = load_model(config.MODEL_PATH)
Y_pred = model.predict_generator(test_generator)
y_pred = np.argmax(Y_pred, axis=1)


# Generate Confusion Matrix & Classification Report

conf_mat = confusion_matrix(test_generator.classes, y_pred)
target_names = ["drawings", "hentai", "neutral", "porn", "sexy"]

sns_plot = sns.heatmap(
    conf_mat,
    annot=True,
    xticklabels=target_names,
    yticklabels=target_names,
    square=True,
)
sns_plot.figure.savefig("reports/confusion_matrix.png", dpi=300, transparent=True)

report = classification_report(
    test_generator.classes, y_pred, target_names=target_names, output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("reports/classification_report.csv")


# Generate ROC-AUC Curve

fig, c_ax = plt.subplots(1, 1, figsize=(12, 8))


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target_names):
        fpr, tpr, thresholds = roc_curve(y_test[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label="%s (AUC:%0.2f)" % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, "b-", label="Random Guessing")
    c_ax.legend(loc="best")
    c_ax.figure.savefig("reports/roc_auc_curve.png", dpi=300, transparent=True)
    return roc_auc_score(y_test, y_pred, average=average)


print(multiclass_roc_auc_score(test_generator.classes, y_pred))
