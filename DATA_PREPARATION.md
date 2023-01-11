# DATA PREPARATION

In the paper, we provide results for the following datasets:
* Something-Something v2, the few-shot split  (todo add link)
* Kinetics-100, the few-shot split  (todo add link)
* UCF101, the few-shot split (todo add link)

## Download the few-shot splits 

You can follow the 


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

## Save features from the classification model

### Download pre-trained classification models

### Train the classification models

### Save the features

To extract the features, you need:
* pre-trained classification models
* the directory containing all the frames of the videos extracted

#### Download the pre-trained classification models
First, you need to download the pre-trained models. Run
```
TSL_MODELS_DIR=<your_path>

wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_kinetics.pth
wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_ssv2_no_hf.pth
wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_ucf101.pth
```

#### Extract the features
You need to specify:
* ROOT_REPO_DIR (as defined in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md))
* TSL_MODELS_DIR (as defined before)
* VIDEO_DIR
* SPLIT (between train/val/test)
* DATASET (between ssv2/kinetics/ucf101)


```
ROOT_REPO_DIR=<your_repo_dir>
TSL_MODELS_DIR=<your_path>
VIDEO_DIR=<your_path>  # input video frames directory
FEATURE_DIR=<your_path>  # output feature directory
SPLIT=test
DATASET=kinetics

LOG_DIR=${ROOT_REPO_DIR}/feature_extraction_logs
mkdir ${LOG_DIR}
PRETRAIN_DIR=${TSL_MODELS_DIR}/r21d34_pretrained_sports1m_trained_kinetics.pth

TEMPORAL_MATCHING_REPO_DIR=${ROOT_REPO_DIR}/temporal_matching
cd ${TEMPORAL_MATCHING_REPO_DIR}
source ENV/bin/activate

python -u feature_extraction.py \
--input_dataset_dir ${VIDEO_DIR} \
--output_dataset_dir ${FEATURE_DIR} \
--log_dir ${LOG_DIR} \
--dataset ${DATASET} \
--split ${SPLIT} \
--manual_seed 5 \
--pretrain_path ${PRETRAIN_DIR} \
--n_threads 16 \
--r2plus1d_n_classes_pretrain 64 
```
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
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/ssv2/extract_feature_train.sh">script_train</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/ssv2/extract_feature_val.sh">script_val</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/ssv2/extract_feature_test.sh">script_test</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/kinetics/extract_feature_train.sh">script_train</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/kinetics/extract_feature_val.sh">script_val</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/kinetics/extract_feature_test.sh">script_test</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
            <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/ucf101/extract_feature_train.sh">script_train</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/ucf101/extract_feature_val.sh">script_val</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/feature_extraction/ucf101/extract_feature_test.sh">script_test</a></td>
    </tr>
  </tbody>
</table>

## Saved test episodes

For reproducibility, we pre-saved the 10k test episodes for the datasets:
* Something-Something v2, the few-shot split
* Kinetics-100, the few-shot split
* UCF101, the few-shot split




You can download them using the following script.

```
ROOT_TEST_EPISODE_DIR=<your_path>

for DATASET in "ucf101"
do
    DATASET_TEST_EPISODE_DIR=${ROOT_TEST_EPISODE_DIR}/${DATASET}
    mkdir ${DATASET_TEST_EPISODE_DIR}
    cd ${DATASET_TEST_EPISODE_DIR}
    
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/${DATASET}/${DATASET}_w5_s1.tar.gz
    tar -xzf ${DATASET}_w5_s1.tar.gz
    rm -r ${DATASET}_w5_s1.tar.gz
    
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/${DATASET}/${DATASET}_w5_s5.tar.gz
    tar -xzf ${DATASET}_w5_s5.tar.gz
    rm -r ${DATASET}_w5_s5.tar.gz
done
```


The following table provides you the specifications of each dataset.
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
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/ssv2/features/ssv2_w5_s1.tar.gz">ssv2_w5_s1</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/ssv2/features/ssv2_w5_s5.tar.gz">ssv2_w5_s5</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/kinetics/features/kinetics_w5_s1.tar.gz">kinetics_w5_s1</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/kinetics/features/kinetics_w5_s5.tar.gz">kinetics_w5_s5</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/ucf101/features/ucf101_w5_s1.tar.gz">ucf101_w5_s1</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/ucf101/features/ucf101_w5_s5.tar.gz">ucf101_w5_s5</a></td>
    </tr>

  </tbody>
</table>
