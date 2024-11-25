# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:21:36 2024

@author: tang
"""

import glob
from matplotlib import pyplot as plt
import yaml
def configuration(result):
    IMG_DIR = result['IMG_DIR']
    JSON = result['JSON']
    IMG_SIZE_H_ori, IMG_SIZE_W_ori = result['IMG_SIZE_H_ori'],result['IMG_SIZE_W_ori'] #
    global_scale = result['global_scale']
    IMG_SIZE_H, IMG_SIZE_W= result['IMG_SIZE_H'], result['IMG_SIZE_W'] #int(IMG_SIZE_H_ori * global_scale), int(IMG_SIZE_W_ori * global_scale)
    channels = result['channels']
    
    initial_weight = result['initial_weight']
    
    shuffle_num = result['shuffle_num']
    
    BATCH_SIZE = result['BATCH_SIZE']
    variation = eval(result['variation'])
    initial_learning_rate = eval(result['initial_learning_rate'])
    alpha = eval(result['alpha'])
    EPOCHS = result['EPOCHS']
    WARMUP_EPOCHS = result['WARMUP_EPOCHS']
    TrainingFraction = result['TrainingFraction']
    Tranfer_LR = eval(result['Tranfer_LR'])
    early_stop = result['early_stop']
    
    NUM_KEYPOINT = result['NUM_KEYPOINT']
    NUM_KEYPOINTS = result['NUM_KEYPOINT'] * 2
    # Tranfer_EPOCH = 50
    centre = result['centre']
    delta = result['delta']
    bodyparts = result['bodyparts']
    kp_con = [{'name': i, 'bodypart': eval(skeleton)} for i, skeleton in enumerate(result['skeleton'])]
    return IMG_SIZE_H_ori, IMG_SIZE_W_ori, global_scale, IMG_SIZE_H, IMG_SIZE_W, BATCH_SIZE, variation, delta, initial_learning_rate, alpha,EPOCHS, WARMUP_EPOCHS, NUM_KEYPOINT, NUM_KEYPOINTS, shuffle_num, TrainingFraction, Tranfer_LR, channels,IMG_DIR, JSON, kp_con, initial_weight, bodyparts, early_stop,centre