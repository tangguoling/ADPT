# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:49:52 2024

@author: tang
"""

import warnings
warnings.filterwarnings('ignore')
from core.train import train
from core.data_aug import data_augmentation
import yaml
from config.config_training import configuration
import numpy as np
if __name__ == '__main__':
    json_file = 'config.yaml'
    print('\nWellcome to use ADPT v1.2.1 for keypoints detection.')
    print('\nTraining configuration:')
    with open(json_file, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    IMG_SIZE_H_ori, IMG_SIZE_W_ori, global_scale, IMG_SIZE_H, IMG_SIZE_W, BATCH_SIZE, variation, delta, initial_learning_rate, alpha,EPOCHS, WARMUP_EPOCHS, NUM_KEYPOINT, NUM_KEYPOINTS, shuffle_num, TrainingFraction, Tranfer_LR, channels,IMG_DIR, JSON, kp_con, initial_weight, bodyparts, early_stop,centre, num_classes = configuration(result)
    print(result)
    # centre = 4
    # num_classes = 1
    stride = 8
    evaluate = False # T
    # shuffle_num = 3
    save_path = '_' + str(shuffle_num)
    animal = 'singel_mouse'
    model_rmse = train('ADPT',animal, save_path, IMG_SIZE_H_ori, IMG_SIZE_W_ori, global_scale, IMG_SIZE_H, IMG_SIZE_W, BATCH_SIZE, variation, delta, initial_learning_rate, alpha,EPOCHS, WARMUP_EPOCHS, NUM_KEYPOINT, NUM_KEYPOINTS, shuffle_num, TrainingFraction, Tranfer_LR, channels,IMG_DIR, JSON, kp_con, initial_weight, bodyparts,data_augmentation(), early_stop, evaluate,num_classes,centre)
    
    for idx, bodypart in enumerate(bodyparts):
        print('RMSE (' + bodypart + '): ', model_rmse[0][idx])
    print(np.mean(model_rmse[0][:len(bodyparts)]))
    # print('RMSE (average): ', model_rmse[1][:len(bodyparts)])
    
    print('\nNow the model has been trained, and you can start analyzing the videos!')
    
