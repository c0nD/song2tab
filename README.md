# song2tab
Senior Capstone Project by Colin Tiller @ Appalachian State University
## Status: **WIP**

This project is intended to take any form of audio file (containing only guitar) and turn it into a tablature/sheet music
format for guitars.


## Setup
The given dataset being used can be found in the annotations section. Due to the large
size of this dataset, I cannot include it in this repo and you must download and extract
each containing folder to `dataprocessing/data` if you wish to replicate the results/modify
this program yourself. If you do not wish to manually install and move this dataset, the
`process_guitarset.py` file should automatically detect the missing files and download them
for you.


Otherwise, the pre-trained models will be included in the repo, and can be used without needing
to install the original training dataset.


## Annotations
- [Dataset](https://guitarset.weebly.com/)
- [FretNet](https://arxiv.org/abs/2212.03023)
- [BasicPitch](https://arxiv.org/pdf/2203.09893v2.pdf)
- [AMTTools](https://github.com/cwitkowitz/amt-tools)
- [jams](https://github.com/marl/jams)
