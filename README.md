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

for DATASET in "ssv2" "ucf101"
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


## Model Zoo
Model zoo

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
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ssv2</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>1</th>
      <th>66.6 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_1shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_1shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_1shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/config_ssv2_linear_5way_1shots_all_seeds.txt">config</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>5</th>
      <th>80.1 +- 0.3</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_5shots_seed1.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_5shots_seed5.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/ssv2_linear_5way_5shots_seed10.pt">model</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/ssv2/linear/config_ssv2_linear_5way_5shots_all_seeds.txt">config</a></td>
    </tr>
  </tbody>
</table>