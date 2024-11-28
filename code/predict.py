# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 18:49:52 2024

@author: tang
"""

import warnings
warnings.filterwarnings('ignore')
from core.predict import predict, predict_picture
import yaml
from config.config_training import configuration
from config.config_predicting import configuration_predict
if __name__ == '__main__':
    json_file = 'config.yaml'
    print('\nWellcome to use ADPT v1.2.1 for keypoints detection.')
    with open(json_file, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open('config_predict.yaml', 'r', encoding='utf-8') as f:
        result_predict = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    # print('\nTraining configuration:')
    IMG_SIZE_H_ori, IMG_SIZE_W_ori, global_scale, IMG_SIZE_H, IMG_SIZE_W, BATCH_SIZE, variation, delta, initial_learning_rate, alpha,EPOCHS, WARMUP_EPOCHS, NUM_KEYPOINT, NUM_KEYPOINTS, shuffle_num, TrainingFraction, Tranfer_LR, channels,IMG_DIR, JSON, kp_con, initial_weight, bodyparts, early_stop,centre,num_classes = configuration(result)
    # print(result)
    
    videos,save_video,model_path,colors,pcutoff,scorer = configuration_predict(result_predict)
    # centre = 4
    # num_classes = 2 
    stride = 8
    evaluate = False # False
    save_path = '_' + str(shuffle_num)
    
    print('\nStart analyzing videos!\n')
    predict(IMG_SIZE_H_ori, IMG_SIZE_W_ori, global_scale, IMG_SIZE_H, IMG_SIZE_W, BATCH_SIZE, variation, delta, initial_learning_rate, alpha,EPOCHS, WARMUP_EPOCHS, NUM_KEYPOINT, NUM_KEYPOINTS, shuffle_num, TrainingFraction, Tranfer_LR, channels,IMG_DIR, JSON, kp_con, initial_weight, bodyparts,videos,save_video,model_path,colors,pcutoff,num_classes,scorer,centre)
    
    # for idx, bodypart in enumerate(bodyparts):
    #     print('RMSE (' + bodypart + '): ', model_rmse[1][idx])
    # print('RMSE (average): ', model_rmse[1][:len(bodyparts)])
    
    print('\nAll videos has been analyzing!')
