# Model zoo

We saved the pretrained models evaluated in the paper:
* our method: Chamfer++
* prior works:
  * [Generalized Few-Shot Video Classification with Video Retrieval and Feature Generation](https://arxiv.org/pdf/2007.04755.pdf) (TSL)
  * [Temporal-Relational CrossTransformers for Few-Shot Action Recognition](https://arxiv.org/abs/2101.06184) (TRX)
  * [Few-Shot Video Classification via Temporal Alignment](https://arxiv.org/abs/1906.11415) (TSL)
  * [ViSiL: Fine-grained Spatio-Temporal Video Similarity Learning](https://arxiv.org/abs/1908.07410) (ViSiL) adapted for few-shot-action-recognition
* useful baselines:
  * mean
  * max
  * diagonal
  * linear

The following Table recaps the results and scripts for downloading the matching models. 
There is one model per seed (1/5/10).

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Matching method</th>
      <th>Backbone</th>
      <th>Accuracy 1-shot</th>
      <th>Accuracy 5-shots</th>
      <th>Download script</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ssv2</th>
      <th>TSL</th>
      <th>R2+1D</th>
      <th>60.6 +- 0.1</th>
      <th>79.9 +- 0.0</th>
      <th>N/A</th>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>65.8 +- 0.0</th>
      <th>79.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/download_ssv2_mean_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>65.0 +- 0.2</th>
      <th>79.0 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/download_ssv2_max_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>chamfer++</th>
      <th>R2+1D</th>
      <th>67.8 +- 0.2</th>
      <th>81.6 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/download_ssv2_chamfer++_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>66.7 +- 0.1</th>
      <th>80.1 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/download_ssv2_diag_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>66.6 +- 0.1</th>
      <th>80.1 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/download_ssv2_linear_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>OTAM</th>
      <th>R2+1D</th>
      <th>67.1 +- 0.0</th>
      <th>80.2 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/download_ssv2_otam_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>TRX</th>
      <th>R2+1D</th>
      <th>65.5 +- 0.1</th>
      <th>81.8 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/download_ssv2_trx_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>ViSiL</th>
      <th>R2+1D</th>
      <th>67.7 +- 0.0</th>
      <th>81.3 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/download_ssv2_visil_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>TSL</th>
      <th>R2+1D</th>
      <th>93.6 +- 0.0</th>
      <th>98.0 +- 0.0</th>
      <th>N/A</th>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>95.5 +- 0.0</th>
      <th>98.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/mean/download_kinetics_mean_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>95.3 +- 0.1</th>
      <th>98.3 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/max/download_kinetics_max_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>chamfer++</th>
      <th>R2+1D</th>
      <th>96.1 +- 0.1</th>
      <th>98.3 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/chamfer++/download_kinetics_chamfer++_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>95.3 +- 0.1</th>
      <th>98.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/diag/download_kinetics_diag_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>95.5 +- 0.1</th>
      <th>98.1 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/linear/download_kinetics_linear_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>OTAM</th>
      <th>R2+1D</th>
      <th>95.9 +- 0.0</th>
      <th>98.4 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/otam/download_kinetics_otam_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>TRX</th>
      <th>R2+1D</th>
      <th>93.4 +- 0.2</th>
      <th>97.5 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/trx/download_kinetics_trx_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>ViSiL</th>
      <th>R2+1D</th>
      <th>95.9 +- 0.0</th>
      <th>98.2 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/visil/download_kinetics_visil_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>TSL</th>
      <th>R2+1D</th>
      <th>97.1 +- 0.0</th>
      <th>99.4 +- 0.0</th>
      <th>N/A</th>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>97.6 +- 0.2</th>
      <th>98.9 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/mean/download_ucf101_mean_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>97.9 +- 0.1</th>
      <th>98.9 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/max/download_ucf101_max_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>chamfer++</th>
      <th>R2+1D</th>
      <th>97.7 +- 0.0</th>
      <th>99.3 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/chamfer++/download_ucf101_chamfer++_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>97.6 +- 0.2</th>
      <th>99.0 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/diag/download_ucf101_diag_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>97.6 +- 0.1</th>
      <th>98.9 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/linear/download_ucf101_linear_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>OTAM</th>
      <th>R2+1D</th>
      <th>97.8 +- 0.1</th>
      <th>99.0 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/otam/download_ucf101_otam_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>TRX</th>
      <th>R2+1D</th>
      <th>96.6 +- 0.0</th>
      <th>99.5 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/trx/download_ucf101_trx_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>ViSiL</th>
      <th>R2+1D</th>
      <th>97.8 +- 0.2</th>
      <th>99.0 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/visil/download_ucf101_visil_5way_all_shots_all_seeds.txt">download</a></td>
    </tr>
  </tbody>
</table>

To download the pre-trained models for a given matching function, you need to specify:
* CHECKPOINT_DIR, where the models will be saved
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* DATASET (between ssv2/kinetics/ucf101)

and then run

```
CHECKPOINT_DIR=<your_checkpoint_dir>
MATCHING_NAME=chamfer++
DATASET=ssv2
cd ${CHECKPOINT_DIR}
for SHOT in 1 5
do
    for SEED in 1 5 10
    do
      wget http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/models/matching/${DATASET}/${MATCHING_NAME}/${DATASET}_${MATCHING_NAME}_5way_${SHOT}shots_seed${SEED}.pt
    done
done
```


Note that the classifier-based approach doesn't use matching models, and train a new classifier 
for each test episode.
