# NSFW Classifier

## Problem Statement
To make browsing web and popular social media platforms like Facebook, Instagram, Twitter, Tumblr more safer for children by curbing down on NSFW content present in the form of Images & Videos.

## Business Problem
Of all the content present on the Internet, 30% contains pornography and much more contains NSFW content. With abundance of such content, it is required to build a tool which can filter floating images and videos so as build a safer community and web experience. 

## Data Overview
We made use of [NSFW Data Scraper](https://github.com/alex000kim/nsfw_data_scraper) in order to construct the entire dataset. This repository comprises a set of scripts that allows for an automatic collection of tens of thousands of images for the following (loosely defined) categories to be later used for training an image classifier:

- **porn** - pornography images
- **hentai** - hentai images, but also includes pornographic drawings
- **sexy** - sexually explicit images, but not pornography. Think nude photos, playboy, bikini, etc.
- **neutral** - safe for work neutral images of everyday things and people
- **drawings** - safe for work drawings (including anime)

After collecting all the images and removing the corrupted ones, we ended up with roughly 74k total images which accounted for approximately 34GBs of disk space. For the first experiment, we made use of a subset of this entire dataset which contained roughly 30k images (20k for training and 10k for testing)and occupied 12GBs of disk space. The following table depicts the data structure:

| Categories | Train | Test |
| ------ | ------ | ------ |
| Drawing | 2000 | 2000 |
| Hentai | 2000 | 2000 |
| Neutral | 8000 | 2000 |
| Porn | 3000 | 2000 |
| Sexy | 5000 | 2000 |
| **Total** | **20000** | **10000** |

## Business Objectives & Constraints
- **Low Latency Requirement:** While browsing web, we wanted the model to make inference as quickly as possible so that there isn't any lag in content getting identified which might ultimately lead to bad user experience.
- **Confidence Probability:** Model should be able to predict confidence probability scores for each prediction being made.
- **High Accuracy:** Model should be able to identify NFSW images & ignore the safe ones with high accuracy.

## Performance Metrics
- High Precision & High Recall
- Confusion Matrix

## Techniques
We followed Supervised Learning paradigm from the realm of Machine Learning. As the data we worked upon was unstructured in nature i.e. images/videos, it made more sense to go with Deep Learning based models like Convolutional Neural Nets. Although, we had good amount of data to work with, but knowing that the ImageNet contains some NSFW images, we used pretrained architectures and later fine-tuned over images in our dataset. We made use of the following models pretrained on ImageNet:

- [MobileNetV2](https://keras.io/api/applications/mobilenet/) Smaller Weights, Faster Loading & Inference, Decently Accurate
- [NASMobileNet](https://keras.io/api/applications/nasnet/#nasnetmobile-function) Smaller Weights, Faster Loading & Inference, Decently Accurate
- [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function) Comparably Larger Weights, Comparably Slower Loading, Better Accuracy

## Experimentation & Validation
We trained aforementioned architectures over the training dataset spanning over 5 classes and comprising of roughly 20k images. Following are the transformations performed on the training images:

```python
rescale=1./255,
rotation_range=30,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
channel_shift_range=20,
horizontal_flip=True
```
Following are model setup details:
```python
image_size=224
epochs=100
steps=500
```
Following is the model-head added to the aforementioned base architectures (MobileNet/NASMobileNet/EfficientNetB0):
```python
base model
⬇️
AveragePooling2D with pooling size (7, 7)
⬇️
Flatten
⬇️
Dense(32) with ReLU
⬇️
Batch Normalization
⬇️
Dropout (0.5)
⬇️
Dense(5) with SoftMax
```
Later, we validated our model using images in our testing set. We didn't perform any transformation other than normalizing/rescaling them. 

```python
rescale=1./255
```

## Results
- **Confusion Matrix**
- **Classification Report**
- **ROC-AUC Curve**

## Pitfalls

Although, the model gives good enough results but still suffers from few issues. 
- Sometimes the model gets confused between sexy and porn images. There are images which could be either of them yet one of them is predicted and that too with low confidence scores. Possible reason could be due to the training dataset being noisy, the model trained upon it, is sometimes unsure and unconfident of its predictions. 

- The model is not able to classify those porn images as porn which contain male genitalia predominantly. Possible reason could be that the majority of the porn images are female dominant leading to the males not being centric for featurization to take advantage of.

- Sometimes models predict the input image to be either hentai or drawing interchangeably. Possible reasons could be due to low representation and not so much difference between hentai and drawing categories.

## Solutions

We believe following could help us mitigate some of the above mentioned issues:

- Less noisy dataset.

- More representation for hentai and drawing categories.

- Using a separate classifier model attributed to label an image to be containing female and male body parts for better combined confidence score. 


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
