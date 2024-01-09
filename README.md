# Anti-Drift Pose Tracker (ADPT): A transformer-based network for robust animal pose estimation cross-species

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Key Findings

The analysis results demonstrate that ADPT significantly reduces body point drifting in animal pose estimation and outperforms existing deep learning methods such as DeepLabCut, SLEAP, and DeepPoseKit. Additionally, ADPT's anti-drift tracking is unbiased across different individuals and video background conditions, ensuring consistency for subsequent behavioral analyses. In performance evaluations on public datasets, ADPT exhibited higher accuracy in tracking body points, superior performance metrics in terms of required training data, and inference speed.

Furthermore, our team applied ADPT to end-to-end multi-animal identity-pose synchronized tracking, achieving over 90% accuracy in identity recognition. This end-to-end approach reduces computational costs compared to methods like ma-DLC, SIPEC, and Social Behavior Atlas, potentially enabling real-time multi-animal behavior analysis.

## Usage

This tool provides Python scripts for training and predicting behavior videos. Users can simply open the corresponding environment and run the provided code to start training and predicting.

### Training
- step 1 Modify config.yaml.
- step 2 Open a terminal and enter the folder where the train.py is located.
- step 3 Run train.py and wait for the training to complete.
```bash
python train.py
```
- step 4 The trained model will be saved in the subfolder (model/ADPT/_shuffle_num/cp.ckpt)

### Dependencies

- python 3.9
- tensorflow 2.9.1
- tensorflow-addons 0.17.1
- cudnn
- imgaug
- opencv
- matplotlib

### Data Preparation

During the data preparation phase, we do not provide specific annotation tools. Users can employ the DeepLabCut tool for data annotation and convert the annotations into our required JSON dataset using the [provided code](data/dlc2adpt.py).

### Data 

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE.txt) file for details.
