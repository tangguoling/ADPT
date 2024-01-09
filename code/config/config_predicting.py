# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:45:08 2024

@author: tang
"""

import glob

def configuration_predict(result_predict):
    videos = glob.glob(result_predict['Video_path']+ '/*.' + result_predict['Video_type'])
    save_video = result_predict['save_predicted_video']
    # shuffle_num = 10086
    model_path = result_predict['model_path']
    colors = [eval(color) for color in result_predict['colors']]
    pcutoff = result_predict['pcutoff']
    return videos,save_video,model_path,colors,pcutoff