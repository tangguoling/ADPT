# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:36:10 2022

@author: tang
"""
import cv2
import glob
import pandas as pd
import json
import h5py
csvs = glob.glob('D:/ADPT/data/mouse/labeled-data/*/CollectedData_*.csv')
f = open('mouse.json','w')

all_data = []
for csv in csvs:
    data = pd.read_csv(csv)
    
    data = pd.read_csv(csv).values
    num_picture = len(data)
    for i in range(2,num_picture):
        
        img_path = data[i,1] +'/' +data[i,2]
        keypoints_x = data[i,3::2]
        keypoints_y = data[i,4::2]
        
        joints = []
        for i in range(len(keypoints_x)):
            if keypoints_x[i] != 'nan':
                keypoints_x[i] = float(keypoints_x[i])
            else:
                keypoints_x[i] =  0
            if keypoints_y[i] != 'nan':
                keypoints_y[i] = float(keypoints_y[i])
            else:
                keypoints_y[i] =  0
            x = keypoints_x[i]
            y = keypoints_y[i]
            joints.append([x,y,1])
        
        min_x = min(keypoints_x)-20
        min_y = min(keypoints_y)-20
        max_x = max(keypoints_x)+20
        max_y = max(keypoints_y)+20
        width = max_x - min_x
        height = max_y - min_y
        img_bbox = [min_x,min_y,width,height]
        all_data.append({
            "img_path":img_path,
            "joints":joints,"img_bbox":img_bbox})
json.dump(all_data, f)
f.close()