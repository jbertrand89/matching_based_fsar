# DATA PREPARATION

In the paper, we provide results for the following datasets:
* [Something-Something v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [Kinetics](https://www.deepmind.com/open-source/kinetics)
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

## Few-shot datasets

First you need to download the videos and extract the video frames.
We use the few-shot splits (train/val/test) from 
[TSL](https://github.com/xianyongqin/few-shot-video-classification/data), to be able to train and
evaluate the 64-classes classifier in a similar fashion.

Because the kinetics dataset may change over time (videos are continually removed from 
youtube/marked as private), I saved the video frames extracted. You can download it using the 
following script:

<details>
  <summary> <b> Code </b> </summary>

```
VIDEO_FRAMES_DIR=<your_path>
mkdir ${VIDEO_FRAMES_DIR}
DATASET=kinetics
VIDEO_FRAMES_DATASET_DIR=${VIDEO_FRAMES_DIR}/${DATASET}
mkdir ${VIDEO_FRAMES_DATASET_DIR}
cd ${VIDEO_FRAMES_DATASET_DIR}

for SPLIT in train val test classification_val
do
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/few_shot_splits/kinetics/${SPLIT}.tar.gz
    tar -xzf ${SPLIT}.tar.gz
    rm -r ${SPLIT}.tar.gz
done
```
</details>


## R(2+1)D features

To compare fairly classifier-based and matching-based approaches, we start from frozen R(2+1)D 
features. We extract the features for each 16-frames clip at each temporal position in the video.
Clips are spatially augmented during feature extraction for the train split only:
  * a random cropping is applied to every frame of a clip, 
  * random horizontal flipping is applied for Kinetics and UCF101 datasets. 

For the val and test splits, no spatial augmentation (no crop and no random horizontal flipping)
are applied during feature extraction.
  
To extract the features, you need:
* pre-trained [feature models](#download-the-feature-models) 
* the directory containing all the video frames, as defined in [Few-shot-datasets](#few-shot-datasets) 

### Download the feature models
First, download the pre-trained models by running
<details>
  <summary> <b> Code </b> </summary>

```
FEATURE_MODELS_DIR=<your_path>
mkdir ${FEATURE_MODELS_DIR}
cd ${FEATURE_MODELS_DIR}

wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_kinetics.pth
wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_ssv2_no_hf.pth
wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_ucf101.pth
```
</details>


### Feature extraction
You need to specify:
* ROOT_REPO_DIR (as defined in [Installation](https://github.com/jbertrand89/temporal_matching/#installation))
* FEATURE_MODELS_DIR (as defined in [Download the feature models](#download-the-feature-models))
* VIDEO_FRAMES_DIR (as defined in [Few-shot-datasets](#few-shot-datasets) )
* SPLIT (between train/val/test)
* DATASET (between ssv2/kinetics/ucf101)

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_REPO_DIR=<your_repo_dir>
FEATURE_MODELS_DIR=<your_path>
VIDEO_FRAMES_DIR=<your_path>  # input video frames directory
FEATURE_DIR=<your_path>  # output feature directory
SPLIT=test
DATASET=kinetics

VIDEO_FRAMES_DATASET_DIR=${VIDEO_FRAMES_DIR}/${DATASET}
FEATURE_DATASET_DIR=${FEATURE_DIR}/${DATASET}
mkdir ${FEATURE_DATASET_DIR}
LOG_DIR=${ROOT_REPO_DIR}/feature_extraction_logs
mkdir ${LOG_DIR}
PRETRAIN_FILENAME=${FEATURE_MODELS_DIR}/r21d34_pretrained_sports1m_trained_kinetics.pth

TEMPORAL_MATCHING_REPO_DIR=${ROOT_REPO_DIR}/temporal_matching
cd ${TEMPORAL_MATCHING_REPO_DIR}
source ENV/bin/activate

python -u feature_extraction.py \
--input_dataset_dir ${VIDEO_FRAMES_DATASET_DIR} \
--output_dataset_dir ${FEATURE_DATASET_DIR} \
--log_dir ${LOG_DIR} \
--dataset ${DATASET} \
--split ${SPLIT} \
--manual_seed 5 \
--pretrain_path ${PRETRAIN_FILENAME} \
--n_threads 16 \
--r2plus1d_n_classes_pretrain 64 
```
</details>

The following table recaps the scripts for corresponding to each few-shot split and each dataset.
<details>
  <summary> <b> Table </b> </summary>

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>script train</th>
      <th>script val</th>
      <th>script test </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ssv2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/ssv2/extract_feature_train.sh">script_train</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/ssv2/extract_feature_val.sh">script_val</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/ssv2/extract_feature_test.sh">script_test</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/kinetics/extract_feature_train.sh">script_train</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/kinetics/extract_feature_val.sh">script_val</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/kinetics/extract_feature_test.sh">script_test</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
            <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/ucf101/extract_feature_train.sh">script_train</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/ucf101/extract_feature_val.sh">script_val</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/feature_extraction/ucf101/extract_feature_test.sh">script_test</a></td>
    </tr>
  </tbody>
</table>
</details>


### Download pre-saved features

For reproducibility, I saved features for the train, test and val few-shot splits.

To download them, you need to specify:
* ROOT_FEATURE_DIR, root directory where to save the features
* DATASET (between ssv2/kinetics/ucf101)

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_FEATURE_DIR=<your_path>
DATASET=ssv2

FEATURE_DATASET_DIR=${ROOT_FEATURE_DIR}/${DATASET}
mkdir ${FEATURE_DATASET_DIR}
cd ${FEATURE_DATASET_DIR}

for SPLIT in val test train
do
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/feature_saved/ssv2/${SPLIT}.tar.gz 
    tar -xzf ${SPLIT}.tar.gz 
    rm -r ${SPLIT}.tar.gz 
done
```
</details>


## Test episodes

For reproducibility, we pre-saved the 10k test episodes that were used in the paper, for each of the
three datasets. 

Each episode contains:
* support features, a tensor containing the R(2+1)D features of the support examples
* support labels, a tensor containing the labels of the support examples
* support frame names, a list of the frame paths to compute each support clip
* query features, a tensor containing the R(2+1)D features of the query examples
* query labels, a tensor containing the labels of the query examples
* query frame names, a list of the frame paths to compute each query clip

To compute features for a different backbone, you can start from the support and query frame names. 
This will enable to fairly compare between different methods using different backbones.

You can download the test episodes using the following script

<details>
  <summary> <b> Code </b> </summary>

```
ROOT_TEST_EPISODE_DIR=<your_path>
mkdir ${ROOT_TEST_EPISODE_DIR}
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