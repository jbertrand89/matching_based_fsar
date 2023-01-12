# Rethinking matching-based few-shot action recognition

Juliette Bertrand, Yannis Kalantidis, Giorgos Tolias

[[arXiv]()] [[project page](https://jbertrand89.github.io/temporal_matching_project_page/)]

This repository contains official code for the above-mentioned publication.


## Installation

Please follow the steps described in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md).


## Data preparation

For more details on the datasets, please refer to [DATA_PREPARATION](https://github.com/jbertrand89/temporal_matching/blob/main/DATA_PREPARATION.md).


## Inference

We saved the pretrained models presented in the paper:
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


### Model zoo

The following Table recaps all the scripts to 
* download models
* run inference on the pre-saved episodes 
* run inference using the dataloader. 

The following subsections describe each steps for the example of the Chamfer++ matching.


<table>
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Matching method</th>
      <th>Backbone</th>
      <th>Accuracy 1-shot</th>
      <th>Accuracy 5-shots</th>
      <th>Download models</th>
      <th>Inference from saved episodes</th>
      <th>Inference from dataloader</th>
      <th>Train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ssv2</th>
      <th>tsl</th>
      <th>R2+1D</th>
      <th>60.6 +- 0.1</th>
      <th>79.9 +- 0.0</th>
      <th>N/A</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/classification/ssv2/inference_ssv2_tsl_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td>N/A</td>
      <th>N/A</th>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>65.8 +- 0.0</th>
      <th>79.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/download_ssv2_mean_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/inference_ssv2_mean_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/inference_loader_ssv2_mean_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/mean/train_ssv2_mean_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>65.0 +- 0.2</th>
      <th>79.0 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/download_ssv2_max_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/inference_ssv2_max_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/inference_loader_ssv2_max_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/max/train_ssv2_max_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>chamfer++</th>
      <th>R2+1D</th>
      <th>67.8 +- 0.2</th>
      <th>81.6 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/download_ssv2_chamfer++_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/inference_ssv2_chamfer++_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/inference_loader_ssv2_chamfer++_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/chamfer++/train_ssv2_chamfer++_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>66.7 +- 0.1</th>
      <th>80.1 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/download_ssv2_diag_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/inference_ssv2_diag_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/inference_loader_ssv2_diag_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/diag/train_ssv2_diag_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>66.6 +- 0.1</th>
      <th>80.1 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/download_ssv2_linear_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/inference_ssv2_linear_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/inference_loader_ssv2_linear_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/linear/train_ssv2_linear_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>otam</th>
      <th>R2+1D</th>
      <th>67.1 +- 0.0</th>
      <th>80.2 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/download_ssv2_otam_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/inference_ssv2_otam_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/inference_loader_ssv2_otam_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/otam/train_ssv2_otam_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>trx</th>
      <th>R2+1D</th>
      <th>65.5 +- 0.1</th>
      <th>81.8 +- 0.2</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/download_ssv2_trx_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/inference_ssv2_trx_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/inference_loader_ssv2_trx_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/trx/train_ssv2_trx_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ssv2</th>
      <th>visil</th>
      <th>R2+1D</th>
      <th>67.7 +- 0.0</th>
      <th>81.3 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/download_ssv2_visil_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/inference_ssv2_visil_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/inference_loader_ssv2_visil_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ssv2/visil/train_ssv2_visil_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>tsl</th>
      <th>R2+1D</th>
      <th>93.6 +- 0.0</th>
      <th>98.0 +- 0.0</th>
      <th>N/A</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/classification/kinetics/inference_kinetics_tsl_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td>N/A</td>
      <th>N/A</th>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>95.5 +- 0.0</th>
      <th>98.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/mean/download_kinetics_mean_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/mean/inference_kinetics_mean_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/mean/inference_loader_kinetics_mean_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/mean/train_kinetics_mean_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>95.3 +- 0.1</th>
      <th>98.3 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/max/download_kinetics_max_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/max/inference_kinetics_max_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/max/inference_loader_kinetics_max_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/max/train_kinetics_max_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>chamfer++</th>
      <th>R2+1D</th>
      <th>96.1 +- 0.1</th>
      <th>98.3 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/chamfer++/download_kinetics_chamfer++_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/chamfer++/inference_kinetics_chamfer++_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/chamfer++/inference_loader_kinetics_chamfer++_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/chamfer++/train_kinetics_chamfer++_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>95.3 +- 0.1</th>
      <th>98.1 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/diag/download_kinetics_diag_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/diag/inference_kinetics_diag_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/diag/inference_loader_kinetics_diag_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/diag/train_kinetics_diag_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>95.5 +- 0.1</th>
      <th>98.1 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/linear/download_kinetics_linear_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/linear/inference_kinetics_linear_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/linear/inference_loader_kinetics_linear_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/linear/train_kinetics_linear_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>otam</th>
      <th>R2+1D</th>
      <th>95.9 +- 0.0</th>
      <th>98.4 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/otam/download_kinetics_otam_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/otam/inference_kinetics_otam_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/otam/inference_loader_kinetics_otam_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/otam/train_kinetics_otam_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>trx</th>
      <th>R2+1D</th>
      <th>93.4 +- 0.2</th>
      <th>97.5 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/trx/download_kinetics_trx_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/trx/inference_kinetics_trx_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/trx/inference_loader_kinetics_trx_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/trx/train_kinetics_trx_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>kinetics</th>
      <th>visil</th>
      <th>R2+1D</th>
      <th>95.9 +- 0.0</th>
      <th>98.2 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/visil/download_kinetics_visil_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/visil/inference_kinetics_visil_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/visil/inference_loader_kinetics_visil_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/kinetics/visil/train_kinetics_visil_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>tsl</th>
      <th>R2+1D</th>
      <th>97.1 +- 0.0</th>
      <th>99.4 +- 0.0</th>
      <th>N/A</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/classification/ucf101/inference_ucf101_tsl_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td>N/A</td>
      <th>N/A</th>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>mean</th>
      <th>R2+1D</th>
      <th>97.6 +- 0.2</th>
      <th>98.9 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/mean/download_ucf101_mean_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/mean/inference_ucf101_mean_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/mean/inference_loader_ucf101_mean_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/mean/train_ucf101_mean_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>max</th>
      <th>R2+1D</th>
      <th>97.9 +- 0.1</th>
      <th>98.9 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/max/download_ucf101_max_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/max/inference_ucf101_max_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/max/inference_loader_ucf101_max_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/max/train_ucf101_max_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>chamfer++</th>
      <th>R2+1D</th>
      <th>97.7 +- 0.0</th>
      <th>99.3 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/chamfer++/download_ucf101_chamfer++_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/chamfer++/inference_ucf101_chamfer++_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/chamfer++/inference_loader_ucf101_chamfer++_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/chamfer++/train_ucf101_chamfer++_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>diagonal</th>
      <th>R2+1D</th>
      <th>97.6 +- 0.2</th>
      <th>99.0 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/diag/download_ucf101_diag_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/diag/inference_ucf101_diag_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/diag/inference_loader_ucf101_diag_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/diag/train_ucf101_diag_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>linear</th>
      <th>R2+1D</th>
      <th>97.6 +- 0.1</th>
      <th>98.9 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/linear/download_ucf101_linear_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/linear/inference_ucf101_linear_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/linear/inference_loader_ucf101_linear_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/linear/train_ucf101_linear_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>otam</th>
      <th>R2+1D</th>
      <th>97.8 +- 0.1</th>
      <th>99.0 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/otam/download_ucf101_otam_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/otam/inference_ucf101_otam_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/otam/inference_loader_ucf101_otam_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/otam/train_ucf101_otam_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>trx</th>
      <th>R2+1D</th>
      <th>96.6 +- 0.0</th>
      <th>99.5 +- 0.0</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/trx/download_ucf101_trx_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/trx/inference_ucf101_trx_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/trx/inference_loader_ucf101_trx_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/trx/train_ucf101_trx_5way_1shots_seed1.txt">train</a></td>
    </tr>
    <tr>
      <th>ucf101</th>
      <th>visil</th>
      <th>R2+1D</th>
      <th>97.8 +- 0.2</th>
      <th>99.0 +- 0.1</th>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/visil/download_ucf101_visil_5way_all_shots_all_seeds.txt">download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/visil/inference_ucf101_visil_5way_1shots_all_seeds.txt">from_episodes</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/visil/inference_loader_ucf101_visil_5way_1shots_all_seeds.txt">from_loader</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/bertrjul/temporal_matching/scripts/matching/ucf101/visil/train_ucf101_visil_5way_1shots_seed1.txt">train</a></td>
    </tr>
  </tbody>
</table>


### Download pre-trained models
To download the pre-trained models for a given matching function, you need to specify:
* CHECKPOINT_DIR, where the models will be saved
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* DATASET (between ssv2/kinetics/ucf101)

and then run
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

### Download pre-saved test episodes

For reproducibility, we pre-saved the 10k test episodes for the datasets:
* Something-Something v2, the few-shot split
* Kinetics-100, the few-shot split
* UCF101, the few-shot split

You can download them using the following script

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


### Run inference for a pre-trained model on the pre-saved episodes

You first need to
* download the pre-trained episodes for each dataset
* download the pre-trained models for multiple seeds 


To run inference for a given matching function on pre-saved episodes, you need to specify:
* ROOT_TEST_EPISODE_DIR
* CHECKPOINT_DIR 
* ROOT_REPO_DIR (as defined in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* SHOT (number of example per class between 1/5)
* DATASET (between ssv2/kinetics/ucf101)

And then run the script. Each script is different depending on the matching function, so please
refer to the model zoo to find the one you need. For example, with Chamfer++ matching, 
run
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


### Run inference for a pre-trained model on any episodes 

You first need to
* download the test pre-saved features (see section training)
* download the pre-trained models for multiple seeds 


To run inference for a given matching function on pre-saved episodes, you need to specify:
* ROOT_FEATURE_DIR
* CHECKPOINT_DIR 
* ROOT_REPO_DIR (as defined in [GETTING_STARTED](https://github.com/jbertrand89/temporal_matching/blob/main/GETTING_STARTED.md))
* MATCHING_NAME (between diag/mean/max/linear/otam/chamfer++/trx/visil)
* SHOT (number of example per class between 1/5)
* DATASET (between ssv2/kinetics/ucf101)
* TEST_SEED (the number you like)

And then run the script. Each script is different depending on the matching function, so please
refer to the model zoo to find the one you need. For example, with Chamfer++ matching, 
run

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


## Training the models

### Download the pre-saved features

You need to specify:
* ROOT_FEATURE_DIR, root directory where to save the features
* DATASET (between ssv2/kinetics/ucf101)

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

### Train a model

To run inference for a given matching function on pre-saved episodes, you need to specify:
* CHECKPOINT_DIR
* ROOT_FEATURE_DIR 
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



