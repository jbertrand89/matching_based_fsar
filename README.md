# Rethinking matching-based few-shot action recognition

[[arXiv]()] [[project page](https://jbertrand89.github.io/temporal_matching_project_page/)]

This repository contains official code for our paper 
[Rethinking matching-based few-shot action recognition](https://jbertrand89.github.io/temporal_matching_project_page/).

## What do we have here?

1. [Installation](#installation)

2. [Data preparation](#data-preparation)

3. [Model zoo](#model-zoo)

4. [Evaluating a pre-trained model](#evaluating-a-pre-trained-model)
   1. [On pre-saved episodes](#on-pre-saved-episodes)
   2. [General use-case](#general-use-case)

5. [Train a model](#train-a-model)

6. [Scripts summary](#scripts-summary)

7. [Citation](#citation)


## Installation

This code is based on the 
[TSL](https://github.com/xianyongqin/few-shot-video-classification) [[1] ](#references) and 
[TRX](https://github.com/tobyperrett/few-shot-action-recognition) [[2] ](#references) repositories. 
It requires Python >= 3.8

You can find below the installation script:

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_REPO_DIR=<path_to_the_root_folder>
cd ${ROOT_REPO_DIR}
git clone https://github.com/xianyongqin/few-shot-video-classification.git
git clone https://github.com/tobyperrett/few-shot-action-recognition.git
git clone https://github.com/jbertrand89/temporal_matching.git
cd temporal_matching

python -m venv ENV
source ENV/bin/activate
pip install torch torchvision==0.12.0
pip install tensorboard
pip install einops
pip install ffmpeg
pip install pandas

or use
pip install -r requirements.txt
```
</details>



## Data preparation

For more details on the datasets, please refer to [DATA_PREPARATION](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md).


## Model zoo

We saved the scripts and the pretrained models evaluated in the paper in 
[MODEL_ZOO](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md).

The following sections detail each step.


## Evaluating a pre-trained model

### On pre-saved episodes

To reproduce the paper numbers, you first need to
* [download the test episodes for each dataset](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md#test-episodes)
* [download the pre-trained models for multiple seeds](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md)


To run inference for a given matching function on pre-saved episodes, you need to specify:
* ROOT_TEST_EPISODE_DIR (as defined in [Download test episodes](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md#test-episodes))
* CHECKPOINT_DIR (as defined in [Model ZOO](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md))
* ROOT_REPO_DIR (as defined in [Installation](#installation))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* SHOT (number of example per class between 1/5)
* DATASET (between ssv2/kinetics/ucf101)

And then run the script. Each script is different depending on the matching function, so please
refer to the model zoo to find the one you need. For example, with Chamfer++ matching 
run
<details>
  <summary> <b> Code </b> </summary>

```
ROOT_TEST_EPISODE_DIR=<your_path>
CHECKPOINT_DIR=<your_checkpoint_dir>
ROOT_REPO_DIR=<your_repo_dir>
MATCHING_NAME=chamfer++
SHOT=1
DATASET=ssv2

TEMPORAL_MATCHING_REPO_DIR=${ROOT_REPO_DIR}/temporal_matching
cd ${TEMPORAL_MATCHING_REPO_DIR}
source ENV/bin/activate  # ENV is the name of the environment

for SEED in 1 5 10
do
  MODEL_NAME=${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}.pt
  
  python run_matching.py \
  --num_gpus 1 \
  --num_workers 1 \
  --backbone r2+1d_fc \
  --feature_projection_dimension 1152 \
  --method matching-based \
  --matching_function chamfer \
  --video_to_class_matching joint \
  --clip_tuple_length 3 \
  --shot ${SHOT} \
  --way 5 \
  -c ${CHECKPOINT_DIR} \
  -r -m ${MODEL_NAME} \
  --load_test_episodes \
  --test_episode_dir ${ROOT_TEST_EPISODE_DIR} \
  --dataset_name ${DATASET}
done

python average_multi_seeds.py --result_dir ${CHECKPOINT_DIR} --result_template ${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed --seeds 1 5 10
```
</details>



### General use-case

You may want to run inference on a new set of episodes. We provide a script to use the 
R(2+1)D feature loader.

You first need to
* [download the test pre-saved features](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md#download-pre-saved-features)
* [download the pre-trained models for multiple seeds](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md)


To run inference for a given matching function, you need to specify:
* ROOT_FEATURE_DIR (as defined in [Download pre-saved features](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md#download-pre-saved-features))
* CHECKPOINT_DIR (as defined in [Model ZOO](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md))
* ROOT_REPO_DIR (as defined in [Installation](#installation))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* SHOT (number of example per class between 1/5)
* DATASET (between ssv2/kinetics/ucf101)
* TEST_SEED (the number you like)

And then run the script. Each script is different depending on the matching function, so please
refer to [scripts summary](#scripts-summary) 
to find the one you need. For example, with Chamfer++ matching, run

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_FEATURE_DIR=<your_path>
CHECKPOINT_DIR=<your_checkpoint_dir>
ROOT_REPO_DIR=<your_repo_dir>
MATCHING_NAME=chamfer++
SHOT=1
DATASET=ssv2
TEST_SEED=1
TEST_DIR=${ROOT_FEATURE_DIR}/${DATASET}/test

TEMPORAL_MATCHING_REPO_DIR=${ROOT_REPO_DIR}/temporal_matching
cd ${TEMPORAL_MATCHING_REPO_DIR}
source ENV/bin/activate # ENV is the name of the environment

for SEED in 1 5 10
do
  MODEL_NAME=${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}.pt
  
  python run_matching.py \
  --num_gpus 1 \ 
  --num_workers 1 \
  --backbone r2+1d_fc \
  --feature_projection_dimension 1152 \
  --method matching-based \
  --matching_function chamfer \
  --video_to_class_matching joint \
  --clip_tuple_length 3 \
  --shot ${SHOT} \
  --way 5  \
  -c ${CHECKPOINT_DIR} \
  -r -m ${MODEL_NAME}\
  --split_dirs  ${TEST_DIR} \
  --split_names test \
  --split_seeds ${TEST_SEED}\
  --dataset_name ${DATASET}
done

python average_multi_seeds.py --result_dir ${CHECKPOINT_DIR} --result_template ${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed --seeds 1 5 10
```
</details>



## Train a model

To compare fairly classifier-based and matching-based approaches, we start from frozen 
[R(2+1)D features](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md#download-pre-saved-features)

To run inference for a given matching function on pre-saved episodes, you need to specify:
* CHECKPOINT_DIR (can be different from the one defined in [Model ZOO](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md))
* ROOT_FEATURE_DIR (as defined in [Download pre-saved features](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md#download-pre-saved-features))
* ROOT_REPO_DIR (as defined in [Installation](#installation))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* DATASET (between ssv2/kinetics/ucf101)
* SHOT (number of example per class between 1/5)
* SEED (the number you like, we chose 1/5/10)

The following hyper-parameters were tuned with optuna, we provide you the optimum value found for 
each method
* LR (usually between 0.01/0.001/0.0001)
* GLOBAL_TEMPERATURE
* TEMPERATURE_WEIGHT

And then run the training script. Each script is different depending on the matching function, so 
please refer to the 
[scripts summary](#scripts-summary) to find the one you need. 
For example, with Chamfer++ matching, run 
<details>
  <summary> <b> Code </b> </summary>

```
CHECKPOINT_DIR=<your_checkpoint_dir>
ROOT_FEATURE_DIR=<your_path>
ROOT_REPO_DIR=<your_repo_dir>

MATCHING_NAME=chamfer++
DATASET=ssv2
SHOT=1
SEED=1
LR=0.001  # hyper parameter tuned with optuna
GLOBAL_TEMPERATURE=100  # hyper parameter tuned with optuna
TEMPERATURE_WEIGHT=0.1  # hyper parameter tuned with optuna

TRAIN_FEATURE_DIR=${ROOT_FEATURE_DIR}/${DATASET}/train
VAL_FEATURE_DIR=${ROOT_FEATURE_DIR}/${DATASET}/val
TEST_FEATURE_DIR=${ROOT_FEATURE_DIR}/${DATASET}/test

MODEL_NAME=${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}
CHECKPOINT_DIR_TRAIN=${CHECKPOINT_DIR}/${MODEL_NAME}
rm -r ${CHECKPOINT_DIR_TRAIN}

TEMPORAL_MATCHING_REPO_DIR=${ROOT_REPO_DIR}/temporal_matching
cd ${TEMPORAL_MATCHING_REPO_DIR}
source ENV/bin/activate # ENV is the name of the environment

python run_matching.py \
--dataset_name ${DATASET} \
--tasks_per_batch 1 \
--num_gpus 1 \
--num_workers 1 \
--shot ${SHOT} \
--way 5 \
--query_per_class 1 \
--num_test_tasks 10000 \
--num_val_tasks 10000 \
-c ${CHECKPOINT_DIR_TRAIN} \
--train_split_dir ${TRAIN_FEATURE_DIR} \
--val_split_dir ${VAL_FEATURE_DIR} \
--test_split_dir ${TEST_FEATURE_DIR} \
--train_seed ${SEED} \
--val_seed ${SEED} \
--test_seed 1 \
--seed ${SEED} \
-lr ${LR} \
--matching_global_temperature ${GLOBAL_TEMPERATURE} \
--matching_global_temperature_fixed \
--matching_temperature_weight ${TEMPERATURE_WEIGHT} \
--backbone r2+1d_fc \
--feature_projection_dimension 1152 \
--method matching-based \
--matching_function chamfer \
--video_to_class_matching joint \
--clip_tuple_length 3
```
</details>


## Scripts summary

The following Table recaps the scripts for evaluating and training the following models:
* our method: Chamfer++
* prior work:
  * TSL [[1] ](#references)
  * TRX [[2] ](#references)
  * OTAM [[3] ](#references)
  * ViSiL [[4] ](#references) adapted for few-shot-action-recognition
* useful baselines:
  * mean
  * max
  * diagonal
  * linear

<details>
  <summary> <b> Table </b> </summary>

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Matching method</th>
      <th>Evaluation on saved episodes</th>
      <th>Evaluation, general case</th>
      <th>Training</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tsl</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/classification/ssv2/inference_ssv2_tsl_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td>N/A</td>
      <th>N/A</th>
    </tr>
    <tr>
      <th>mean</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/inference_ssv2_mean_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/inference_loader_ssv2_mean_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/train_ssv2_mean_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>max</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/inference_ssv2_max_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/inference_loader_ssv2_max_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/train_ssv2_max_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>chamfer++</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/inference_ssv2_chamfer++_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/inference_loader_ssv2_chamfer++_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/train_ssv2_chamfer++_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>diagonal</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/inference_ssv2_diag_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/inference_loader_ssv2_diag_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/train_ssv2_diag_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>linear</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/inference_ssv2_linear_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/inference_loader_ssv2_linear_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/train_ssv2_linear_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>otam</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/inference_ssv2_otam_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/inference_loader_ssv2_otam_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/train_ssv2_otam_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>trx</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/inference_ssv2_trx_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/inference_loader_ssv2_trx_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/train_ssv2_trx_5way_1shots_seed1.sh">train</a></td>
    </tr>
    <tr>
      <th>visil</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/inference_ssv2_visil_5way_1shots_all_seeds.sh">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/inference_loader_ssv2_visil_5way_1shots_all_seeds.sh">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/train_ssv2_visil_5way_1shots_seed1.sh">train</a></td>
    </tr>
  </tbody>
</table>
</details>


## Citation

Coming soon.

## References

[1] Xian et al. [Generalized Few-Shot Video Classification with Video Retrieval and Feature Generation](https://arxiv.org/pdf/2007.04755.pdf) 

[2] Perrett et al. [Temporal-Relational CrossTransformers for Few-Shot Action Recognition](https://arxiv.org/abs/2101.06184)

[3] Cao et al. [Few-Shot Video Classification via Temporal Alignment](https://arxiv.org/abs/1906.11415)

[4] Kordopatis-Zilos et al. [ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning](https://arxiv.org/abs/1908.07410)


