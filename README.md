<p align="center">
  <img alt="GitHub release" src="https://github.com/LaxmanSinghTomar/nsfw-classifier/blob/d0ff19d343f97bd747c8500aafb3220eb314e1ce/NSFW%20Classifier.png">
</p>

<p align = "center">
  <a href="https://github.com/LaxmanSinghTomar/nsfw-classifier/commits/master" target="_blank">
    <img src="https://img.shields.io/github/last-commit/LaxmanSinghTomar/nsfw-classifier?style=flat-square" alt="GitHub last commit">
  </a>

<a href="https://github.com/LaxmanSinghTomar/nsfw-classifier/issues" target="_blank">
  <img src="https://img.shields.io/github/issues/LaxmanSinghTomar/nsfw-classifier?style=flat-square&color=red" alt="GitHub issues">
</a>

<a href="https://github.com/LaxmanSinghTomar/nsfw-classifier/pulls" target="_blank">
  <img src="https://img.shields.io/github/issues-pr/LaxmanSinghTomar/nsfw-classifier?style=flat-square&color=blue" alt="GitHub pull requests">
</a>
  
<a href="https://github.com/LaxmanSinghTomar/nsfw-classifier/forks" target="_blank">
  <img src="https://img.shields.io/github/forks/LaxmanSinghTomar/nsfw-classifier?style=flat-square&color=blue" alt="GitHub forks">
</a>

<a href="  https://github.com/LaxmanSinghTomar/nsfw-classifier/stars" target="_blank">
  <img src="https://img.shields.io/github/stars/LaxmanSinghTomar/nsfw-classifier?style=flat-square&color=blue" alt="GitHub stars">
</a>  
  
</br>

<a href="https://github.com/LaxmanSinghTomar/nsfw-classifierr#contribute" target="_blank">
  <img alt="Contributors" src="https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square">
</a>

<a href="https://standardjs.com" target="_blank">
  <img alt="ESLint" src="https://img.shields.io/badge/code_style-standard-brightgreen.svg?style=flat-square">
</a>

<a href="https://github.com/LaxmanSinghTomar/nsfw-classifier/blob/main/LICENSE" target="_blank">
  <img alt="LICENSE" src="https://img.shields.io/github/license/LaxmanSinghTomar/nsfw-classifier">
<a/>

<a href="https://ctt.ac/eqUgY" target="_blank">
  <img src="https://img.shields.io/twitter/url?style=flat-square&logo=twitter&url=https://ctt.ac/eqUgY" alt="GitHub tweet">
</a>
</p>
<hr>

This repository is dedicated for building a classifier to detect NSFW Images &amp; Videos.

# Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [License](#license)

## Installation

[(Back to Top)](#table-of-contents)

To use this project, first clone the repo on your device using the command given below:

```git init```

```git clone https://github.com/LaxmanSinghTomar/nsfw-classifier.git```

## Usage

[(Back to Top)](#table-of-contents)

Install the required libraries & packages using:

```sh
pip install requirements.txt
```

To download the dataset upon which the model was trained run:

```sh
python src/scripts/data.sh
```

If run successfully, this should create a directory ```data``` in the project directory.

To run a quick demo using an image and a video run:

```sh
python src/scripts/inference.sh
```

To identify whether an image contains NSFW content or not using the default model run:

```sh
python src/inference/inference_image.py [img-path]
```

To identify whether a video is NSFW or not using the default model run:

```sh
python src/inference/inference_video.py [video-path]
```

Output Video is saved in the ```output``` directory.

**Note:** The default trained model is MobileNetv2 which is smaller in size due to which loads quickly and is good for inference.

## Development

[(Back to Top)](#table-of-contents)

<pre>
.
├── LICENSE
├── models                         <- Trained and Serialized Models
├── notebooks                      <- Jupyter Notebook
├── NSFW Classifier.png
├── output                         <- Output for Videos
├── README.md
├── references                     <- Reference Materials to understand Approaches & Solutions
├── reports                        <- Reports & Figures Generated
│   ├── figures
├── requirements.txt               <- Requirements File for reproducing the analysis environment 
└── src
    ├── config.py                  <- Script for Configuration like File Paths, default Model
    ├── inference                  <- Scripts for running an inference on either image/video using trained model
    │   ├── inference_image.py
    │   └── inference_video.py
    ├── models                     <- Scripts to train the ML Models
    │   ├── efficientnet.py
    │   ├── mobilenet.py
    │   └── nasnetmobile.py
    ├── scripts                    <- Scripts to download dataset and run inference on an image/video for Demo
    │   ├── data.sh
    │   └── inference.sh
    └── visualizations             <- Scripts to create exploratory and results oriented visualizations
        └── visualizations.py
</pre>

If you wish to change the default model for predictions i.e. MobileNetv2, change ```MODEL_PATH``` in ```src/config.py``` to the either of the models available in ```models``` directory.

## License

[(Back to top)](#table-of-contents)

[GNU General Public License version 3](https://opensource.org/licenses/GPL-3.0)



