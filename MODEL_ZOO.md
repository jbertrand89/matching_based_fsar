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

The following Table recaps all the scripts to 
* download models
* run inference on pre-saved episodes 
* run inference, general case
* training

<details>
  <summary> <b> Table </b> </summary>

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Matching method</th>
      <th>Backbone</th>
      <th>Accuracy 1-shot</th>
      <th>Accuracy 5-shots</th>
      <th>Download models</th>
      <th>Inference on saved episodes</th>
      <th>Inference, general case</th>
      <th>Training</th>
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
</details>

Please refer to sections
[Inference](https://github.com/jbertrand89/temporal_matching#inference) 
and 
[Training](https://github.com/jbertrand89/temporal_matching#training) 
  for more details.