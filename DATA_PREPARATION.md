# DATA PREPARATION

In the paper, we provide results for the following datasets:
* [Something-Something v2](https://developer.qualcomm.com/software/ai-datasets/something-something)
* [Kinetics](https://www.deepmind.com/open-source/kinetics)
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)


We use the few-shot splits from 
[TSL](https://github.com/xianyongqin/few-shot-video-classification/data), to be able to train and
evaluate the 64-classes classifier in a similar fashion.




## Extract features from the classification model

To extract the features, you need:
* pre-trained classification models
* the directory containing all the frames of the videos extracted

### Download the pre-trained classification models
First, you need to download the pre-trained models. Run
```
TSL_MODELS_DIR=<your_path>

wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_kinetics.pth
wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_ssv2_no_hf.pth
wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/classification_pretraining/r21d34_pretrained_sports1m_trained_ucf101.pth
```

### Extract the features
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

