# Anti-Drift Pose Tracker (ADPT): A transformer-based network for robust animal pose estimation cross-species

[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## Key Findings

The analysis results demonstrate that ADPT significantly reduces body point drifting in animal pose estimation and outperforms existing deep learning methods such as [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), [SLEAP](https://github.com/talmolab/sleap), and [DeepPoseKit](https://github.com/jgraving/DeepPoseKit). Additionally, ADPT's anti-drift tracking is unbiased across different individuals and video background conditions, ensuring consistency for subsequent behavioral analyses. In performance evaluations on public datasets, ADPT exhibited higher accuracy in tracking body points, superior performance metrics in terms of required training data, and inference speed.

Furthermore, our team applied ADPT to end-to-end multi-animal identity-pose synchronized tracking, achieving over 90% accuracy in identity recognition. This end-to-end approach reduces computational costs compared to methods like [ma-DLC](https://github.com/DeepLabCut/DeepLabCut), [SIPEC](https://github.com/SIPEC-Animal-Data-Analysis/SIPEC), and [Social Behavior Atlas](https://github.com/YNCris/SBeA_release), potentially enabling real-time multi-animal behavior analysis.

## Usage

This tool provides Python scripts for training and predicting behavior videos. Users can simply open the corresponding environment and run the provided code to start training and predicting.  You may need an NVIDIA GPU (RTX2080TI or better) and updated drivers.
  
### Configure the ADPT virtual environment
```bash
conda create -n ADPT python==3.9
conda activate ADPT
pip install tensorflow==2.9.1
pip install tensorflow-addons==0.17.1
conda install cudnn==8.2.1
pip install imgaug
pip install pandas
pip install pyyaml
pip install tqdm
```

### Train a model with ADPT
- step 1 Modify config.yaml. You may need to modify the image size information, NUM_KEYPOINT, bodyparts and skeleton to correspond to your project. Model information allows you to control model training details. 
- step 2 Open a terminal and enter the folder where the train.py is located. Please make sure that config.yaml is under the same folder of train.py because script would read training configuration from config.py.
- step 3 Run train.py and wait for the training to complete. It may take hours for training a robust enough model. The model may not have been fully trained when the epoch is less than half of EPOCHS or 100 epochs, please retrain the model.
```bash
python train.py
```
- step 4 The trained model will be saved in the subfolder (model/ADPT/_shuffle_num/cp.ckpt).

### Use ADPT to predict videos
- step 1 Modify config_predict.yaml. You may need to modify the Video_type, Video_path (directory), model_path. Please ensure that the videos to be processed have the same size as the original images during model training. Save_predicted_video allows you to control whether save predicted video (True or False).
- step 2 Open a terminal and enter the folder where the predict.py is located. Please make sure that config.yaml and config_predict.yaml are under the same folder of predict.py because script would read configuration from these files.
- step 3 Run predict.py and wait for the prediction to complete. 
```bash
python predict.py
```
- step 4 The prediction files will be saved in the same paths of videos.

### Data Preparation

During the data preparation phase, we do not provide specific annotation tools. Users can employ the [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) tool for data annotation and convert the annotations into our required JSON dataset using the [provided code](data/dlc2adpt.py).

### Provided Training and Test Data

We offer a shared training [dataset and videos](data/link.md) for testing purposes. These videos have been instrumental in our analysis of the model's anti-drift capabilities, as discussed in our research paper.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE.txt) file for details.
