# Rethinking matching-based few-shot action recognition

Juliette Bertrand, Yannis Kalantidis, Giorgos Tolias

[[arXiv]()] [[project page](https://jbertrand89.github.io/temporal_matching_project_page/)]

This repository contains official code for the above-mentioned publication.

## What do we have here?

1. [Installation](#installation)

2. [Data preparation](#data-preparation)

3. [Model zoo](#model-zoo)

4. [Inference](#inference)
   1. [Download models](#download-models)
   2. [Download test episodes](#download-test-episodes)
   3. [Inference on the pre-saved episodes](#inference-on-the-pre-saved-episodes)
   4. [Inference, general use-case](#inference-general-use-case)

5. [Training](#training)
   1. [Download pre-saved features](#download-pre-saved-featuresw)
   2. [Train a model](#train-a-model)

6. [Citation](#citation)


## Installation

Please follow the steps described in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md).


## Data preparation

For more details on the datasets, please refer to [DATA_PREPARATION](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md).


## Model zoo

We saved the scripts and the pretrained models evaluated in the paper in 
[MODEL_ZOO](https://github.com/jbertrand89/temporal_matching/blob/main/MODEL_ZOO.md).

The following sections detail each step.

## Inference

### Download models
To download the pre-trained models for a given matching function, you need to specify:
* CHECKPOINT_DIR, where the models will be saved
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* DATASET (between ssv2/kinetics/ucf101)

and then run
<details>
  <summary> <b> Code </b> </summary>

```
cd ${CHECKPOINT_DIR}
for SHOT in 1 5
do
    for SEED in 1 5 10
    do
      wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/matching/${DATASET}/${MATCHING_NAME}/${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}.pt
    done
done
```
</details>

Note that the classifier-based approach doesn't use additional models, and train a new classifier 
for each test episode.

### Download test episodes

For reproducibility, we pre-saved the 10k test episodes that were used in the paper, for each of the
following datasets:
* Something-Something v2, the few-shot split
* Kinetics-100, the few-shot split
* UCF101, the few-shot split

Each episode contains:
* support features, a tensor containing the R(2+1)D features of the support examples
* support labels, a tensor containing the labels of the support examples
* query features, a tensor containing the R(2+1)D features of the query examples
* query labels, a tensor containing the labels of the query examples

In addition, we also provide
* the support frame names, a list of the frame paths to compute each support clip
* the query frame names, a list of the frame paths to compute each query clip
which will help to compare fairly between different methods with different backbone.

You can download them using the following script

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_TEST_EPISODE_DIR=<your_path>
cd ${ROOT_TEST_EPISODE_DIR}

for DATASET in ssv2 kinetics ucf101
do
    DATASET_TEST_EPISODE_DIR=${ROOT_TEST_EPISODE_DIR}/${DATASET}
    mkdir ${DATASET_TEST_EPISODE_DIR}
    cd ${DATASET_TEST_EPISODE_DIR}
    
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/${DATASET}/${DATASET}_w5_s1.tar.gz
    tar -xzf ${DATASET}_w5_s1.tar.gz
    rm -r ${DATASET}_w5_s1.tar.gz
    
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/${DATASET}/${DATASET}_w5_s5.tar.gz
    tar -xzf ${DATASET}_w5_s5.tar.gz
    rm -r ${DATASET}_w5_s5.tar.gz
done
```
</details>

The following table provides you the specifications of each dataset.
<details>
  <summary> <b> Table </b> </summary>

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Backbone</th>
      <th>#shot</th>
      <th>#classes (way) </th>
      <th>Episode count</th>
      <th>Episodes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ssv2</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/ssv2/ssv2_w5_s1.tar.gz">ssv2_w5_s1</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/ssv2/ssv2_w5_s5.tar.gz">ssv2_w5_s5</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/kinetics/kinetics_w5_s1.tar.gz">kinetics_w5_s1</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/kinetics/kinetics_w5_s5.tar.gz">kinetics_w5_s5</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/ucf101/ucf101_w5_s1.tar.gz">ucf101_w5_s1</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_episodes/ucf101/ucf101_w5_s5.tar.gz">ucf101_w5_s5</a></td>
    </tr>

  </tbody>
</table>
</details>



### Inference on the pre-saved episodes

To reproduce the paper numbers, you first need to
* [download the test episodes for each dataset](#download-test-episodes)
* [download the pre-trained models for multiple seeds](#download-models)


To run inference for a given matching function on pre-saved episodes, you need to specify:
* ROOT_TEST_EPISODE_DIR (as defined in [Download test episodes](#download-test-episodes))
* CHECKPOINT_DIR (as defined in [Download models](#download-models))
* ROOT_REPO_DIR (as defined in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md))
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
source ENV/bin/activate

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



### Inference, general use-case

You may want to run inference on a new set of episodes. We provide a script to use the 
R(2+1)D feature loader.

You first need to
* [download the test pre-saved features](#download-pre-saved-features)
* [download the pre-trained models for multiple seeds](#download-models)


To run inference for a given matching function on pre-saved episodes, you need to specify:
* ROOT_FEATURE_DIR (as defined in [Download pre-saved features](#download-pre-saved-features))
* CHECKPOINT_DIR (as defined in [Download models](#download-models))
* ROOT_REPO_DIR (as defined in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* SHOT (number of example per class between 1/5)
* DATASET (between ssv2/kinetics/ucf101)
* TEST_SEED (the number you like)

And then run the script. Each script is different depending on the matching function, so please
refer to the model zoo to find the one you need. For example, with Chamfer++ matching, 
run

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
source ENV/bin/activate

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



## Training

To compare fairly classifier-based and matching-based approaches, we start from frozen R(2+1)D 
features trained following the first stage of 
[TSL](https://arxiv.org/pdf/2007.04755.pdf), as described in 
[DATA_PREPARATION](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md).

To train a new model, you need to
* download pre-saved features 
* run training


### Download pre-saved features

We extracted features for the train, test and val few-shot splits and saved them for 
reproducibility.

To download them, you need to specify:
* ROOT_FEATURE_DIR, root directory where to save the features
* DATASET (between ssv2/kinetics/ucf101)

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_FEATURE_DIR=<your_path>
DATASET=ssv2

DATASET_FEATURE_DIR=${ROOT_FEATURE_DIR}/${DATASET}
mkdir ${DATASET_FEATURE_DIR}
cd ${DATASET_FEATURE_DIR}

for SPLIT in val test train
do
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/feature_saved/ssv2/${SPLIT}.tar.gz 
    tar -xzf ${SPLIT}.tar.gz 
    rm -r ${SPLIT}.tar.gz 
done
```
</details>


### Train a model

To run inference for a given matching function on pre-saved episodes, you need to specify:
* CHECKPOINT_DIR (can be different from the one defined in [Download models](#download-models))
* ROOT_FEATURE_DIR (as defined in [Download pre-saved features](#download-pre-saved-features))
* ROOT_REPO_DIR (as defined in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* DATASET (between ssv2/kinetics/ucf101)
* SHOT (number of example per class between 1/5)
* SEED (the number you like, we chose 1/5/10)

The following hyper parameters were tuned with optuna, we provide you the optimum value found for 
each method
* LR (usually between 0.01/0.001/0.0001)
* GLOBAL_TEMPERATURE
* TEMPERATURE_WEIGHT

And then run the training script. Each script is different depending on the matching function, so 
please refer to the model zoo to find the one you need. For example, with Chamfer++ matching, 
run 
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
source ENV/bin/activate

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


## Citation

todo


