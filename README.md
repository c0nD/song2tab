# song2tab
Senior Capstone Project by Colin Tiller @ Appalachian State University
## Status: **Trainable Model Available**

This project is intended to take any form of audio file (containing only guitar) and turn it into a tablature/sheet music
format for guitars.


## Setup
Install and activate the conda environment to ensure you have all the correct dependencies:
```sh
conda env create -f environment.yml -n song2tab_env
conda activate song2tab_env
```

### FFmpeg
[librosa](https://librosa.org/doc/main/install.html#ffmpeg) requires an [https://www.ffmpeg.org/](https://www.ffmpeg.org/) install -- please follow their documentation to install, or use the following:
> Windows: `choco install ffmpeg`
> Linux/Unix/OSX (probably already installed) `sudo apt-get install ffmpeg`
> Mac `brew install ffmpeg`

### Windows extra step:
[libsndfile](http://www.mega-nerd.com/libsndfile/) is required for this to load audio properly -- please install this to use this project.  

The given dataset being used can be found in the annotations section. Due to the large
size of this dataset, I cannot include it in this repo and you must download and extract
each containing folder to `dataprocessing/data` if you wish to replicate the results/modify
this program yourself. If you do not wish to manually install and move this dataset, the
`process_guitarset.py` file should automatically detect the missing files and download them
for you.


Otherwise, the pre-trained models will be included in the repo, and can be used without needing
to install the original training dataset.

## My Contributions
My goal with this project is to improve on already existing ideas within the automatic music
transcription academic space. By using Frank Cwitkowitz's `amt-tools`, the framework to
build a model has been fairly straight-forward.

With that said, my goal is to improve upon how the inputs are transformed through
the TabCNN model. By updating how the VQT works to augment data for a more robust model that
accounts for electric guitars, effects, background noise, etc, while also increasing
the model's complexity with extra layers, I aim to create a stronger, more robust model than the example TabCNN.

Additionally, my goal is to host this back-end model on a web-based front-end for others
to access and get tablature from other websites or audio files.

## Annotations
- [Dataset](https://guitarset.weebly.com/)
- [FretNet](https://arxiv.org/abs/2212.03023)
- [BasicPitch](https://arxiv.org/pdf/2203.09893v2.pdf)
- [AMTTools](https://github.com/cwitkowitz/amt-tools)
- [jams](https://github.com/marl/jams)
