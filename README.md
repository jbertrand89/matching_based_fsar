# README

## Saved test episodes

For more details on the datasets, please refer to [DATA_PREPARATION](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md).

For reproducibility, we pre-saved the 10k test episodes for the datasets:
* Something-Something v2, the few-shot split
* Kinetics-100, the few-shot split
* UCF101, the few-shot split




You can download them using the following script.

```
ROOT_TEST_EPISODE_DIR=<your_path>

for DATASET in "ssv2"
do
    DATASET_TEST_EPISODE_DIR=${ROOT_TEST_EPISODE_DIR}/${DATASET}
    mkdir ${DATASET_TEST_EPISODE_DIR}
    DATASET_TEST_FEATURE_DIR=${DATASET_TEST_EPISODE_DIR}/features
    mkdir ${DATASET_TEST_FEATURE_DIR}
    cd ${DATASET_TEST_FEATURE_DIR}
    
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/${DATASET}/features/${DATASET}_w5_s1.tar.gz
    tar -xzf ${DATASET}_w5_s1.tar.gz
    rm -r ${DATASET}_w5_s1.tar.gz
    
    wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/${DATASET}/features/${DATASET}_w5_s5.tar.gz
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
      <th>kinetics100</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/kinetics100/features/kinetics100_w5_s1.tar.gz">kinetics100_w5_s1</a></td>
    </tr>
    <tr>
      <th>kinetics100</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>5</th>
      <th>10000</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/data/test_examples/kinetics100/features/kinetics100_w5_s5.tar.gz">kinetics100_w5_s5</a></td>
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


## Inference


### Model zoo


<table>
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Matching method</th>
      <th>Backbone</th>
      <th># shot</th>
      <th>Accuracy</th>
      <th>Seed 1</th>
      <th>Seed 5</th>
      <th>Seed 10</th>
      <th>Download models</th>
      <th>Inference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ssv2</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>65.8 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_mean_5way_1shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_mean_5way_1shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_mean_5way_1shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/download_ssv2_mean_5way_1shots_all_seeds.txt">script_download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/inference_ssv2_mean_5way_1shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>79.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_mean_5way_5shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_mean_5way_5shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_mean_5way_5shots_seed10.pt">model</a></td>
      <th> - </th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/inference_ssv2_mean_5way_5shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>65.0 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_max_5way_1shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_max_5way_1shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_max_5way_1shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/download_ssv2_max_5way_1shots_all_seeds.txt">script_download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/inference_ssv2_max_5way_1shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>5</th>
      <th></th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_max_5way_5shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_max_5way_5shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/ssv2_max_5way_5shots_seed10.pt">model</a></td>
      <th> - </th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/max/inference_ssv2_max_5way_5shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>66.7 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_diag_5way_1shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_diag_5way_1shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_diag_5way_1shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/download_ssv2_diag_5way_all_shots_all_seeds.txt">script_download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/inference_ssv2_diag_5way_1shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>80.1 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_diag_5way_5shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_diag_5way_5shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_diag_5way_5shots_seed10.pt">model</a></td>
      <th> - </th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/inference_ssv2_diag_5way_5shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>66.6 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_1shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_1shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_1shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/download_ssv2_linear_5way_all_shots_all_seeds.txt">script_download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/inference_ssv2_linear_5way_1shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>80.1 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_5shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_5shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_5shots_seed10.pt">model</a></td>
      <th> - </th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/inference_ssv2_linear_5way_5shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>otam</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>67.1 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_otam_5way_1shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_otam_5way_1shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_otam_5way_1shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/download_ssv2_otam_5way_all_shots_all_seeds.txt">script_download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/inference_ssv2_otam_5way_1shots_all_seeds.txt">script_inference</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>otam</th>
      <th>R2+1D</th>
      <th>5</th>
      <th></th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_otam_5way_5shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_otam_5way_5shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_otam_5way_5shots_seed10.pt">model</a></td>
      <th> - </th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/inference_ssv2_otam_5way_5shots_all_seeds.txt">script_inference</a></td>
    </tr>
  </tbody>
</table>

### Run inference for a pretrained model on the saved episodes

You need to
* download the pretrained models for multiple seeds (the script does it simultaneously for 1 shot and 5 shots)
* run the inference script either for 1 shot or 5 shots

To illustrate the process, you can find below an example using the linear matching function:

#### Download the pretrained models
```
CHECKPOINT_DIR=<your_checkpoint_dir>
MATCHING_NAME=linear
DATASET=ssv2
cd ${CHECKPOINT_DIR}
for SHOT in 1 5
do
    for SEED in 1 5 10
    do
      wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/${DATASET}/${MATCHING_NAME}/${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}.pt
    done
done
```

#### Inference

```
ROOT_TEST_EPISODE_DIR=<your_path>
CHECKPOINT_DIR=<your_checkpoint_dir>
ROOT_REPO_DIR=<your_repo_dir>
MATCHING_NAME=linear
SHOT=5
DATASET=ssv2

TEMPORAL_MATCHING_REPO_DIR=${ROOT_REPO_DIR}/temporal_matching
cd ${TEMPORAL_MATCHING_REPO_DIR}
source ENV/bin/activate

for SEED in 1 5 10
do
  MODEL_NAME=${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}.pt
  python run_matching.py --num_gpus 1 --num_workers 1 --backbone r2+1d_fc --feature_projection_dimension 1152 --method matching-based --matching_function ${MATCHING_NAME} --shot ${SHOT} --way 5  -c ${CHECKPOINT_DIR} -r -m ${MODEL_NAME}  --load_test_episodes --test_episode_dir ${ROOT_TEST_EPISODE_DIR} --dataset_name ${DATASET}
done

python average_multi_seeds.py --result_dir ${CHECKPOINT_DIR} --result_template ${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed --seeds 1 5 10
```






