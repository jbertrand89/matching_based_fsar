# Installation

This code is based on two other repositories:
* [few-shot-video-classification](https://github.com/xianyongqin/few-shot-video-classification) from the paper [Generalized Few-Shot Video Classification with Video Retrieval and Feature Generation](https://arxiv.org/pdf/2007.04755.pdf) 
* [few-shot-action-recognition](https://github.com/tobyperrett/few-shot-action-recognition) from the paper [Temporal-Relational CrossTransformers for Few-Shot Action Recognition](https://arxiv.org/abs/2101.06184)


It requires Python >= 3.8

You can find below the installation script:

```
ROOT_REPO_DIR=<path_to_the_root_folder>
cd ${ROOT_REPO_DIR}
git clone git@github.com:xianyongqin/few-shot-video-classification.git
git clone git@github.com:tobyperrett/few-shot-action-recognition.git
git clone git@github.com:jbertrand89/temporal_matching.git
cd temporal_matching

python -m venv ENV
source ENV/bin/activate
pip install torch torchvision==0.12.0
pip install tensorboard
pip install einops
pip install ffmpeg
```
