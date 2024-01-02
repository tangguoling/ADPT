# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:36:29 2023

@author: tang
"""

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa
import time
import cv2
import tensorflow_addons as tfa
from PIL import Image
# from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os
# import cv2
# import skimage
# from skimage import io
import glob
from skimage.util import img_as_ubyte
from keras import backend
from keras.applications.resnet import ResNet,stack1


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
def mlp(x, hidden_units, dropout_rate = None):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        if dropout_rate is not None:
            x = layers.Dropout(dropout_rate)(x)
    return x   

def stack_fn(x):
    x = stack1(x, 64, 3, stride1=1, name="conv2")
    x = stack1(x, 128, 4, name="conv3")
    x = stack1(x, 256, 6, name="conv4")
    return stack1(x, 512, 3, stride1=1, name="conv5")

def detail_conv(higher_resolutionx_x,filters, stride = 2):
    higher_resolutionx_x = layers.Conv2D(filters * 2,3,stride, padding="same")(higher_resolutionx_x)
    higher_resolutionx_x = layers.Conv2D(filters,3, padding="same")(higher_resolutionx_x)
    higher_resolutionx_x = layers.BatchNormalization()(higher_resolutionx_x)
    higher_resolutionx_x = layers.Activation('relu')(higher_resolutionx_x)
    return higher_resolutionx_x

def encoder(x,block_filters):
    x = tf.keras.layers.MaxPool2D(
        pool_size=2,
        strides=2,
        padding="same",
    )(x)
    for i in range(2):
        x = tf.keras.layers.Conv2D(
            filters=block_filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=True,
        )(x)
        x = tf.keras.layers.Activation(
            activation='relu'
        )(x)
    return x

def ADPT(num_classes,  multianimal = False):
    n_output_channels = NUM_KEYPOINT * 3 + num_classes * 2
    inputs = layers.Input((IMG_SIZE_H_ori,IMG_SIZE_W_ori, channels))
    # x = tf.keras.applications.imagenet_utils.preprocess_input(inputs, mode = 'torch')
    from tensorflow.keras.applications.resnet import preprocess_input
    
    x = preprocess_input(inputs)
    x = tf.image.resize(x,(IMG_SIZE_H,IMG_SIZE_W))
    backbone = ResNet(
            stack_fn,
            False,
            True,
            "resnet50",
            False,
            'imagenet',
            x
        )
  
    higher_resolutionx_4x = backbone.get_layer("conv2_block3_out").output
    higher_resolutionx_8x = encoder(higher_resolutionx_4x,256)
    x = encoder(higher_resolutionx_8x,512)
    
    higher_resolutionx_4x = detail_conv(higher_resolutionx_4x,128)
    
    patch_dims = x.shape[-1]
    w,h = x.shape[-2],x.shape[-3]
    x = layers.Reshape((-1, patch_dims))(x)
    num_patches = x.shape[-2]
    
    patch_dims_trans = 128
    encoded_patches = PatchEncoder(num_patches, patch_dims_trans)(x)
    # patch_dims = int(patch_dims / 2)
    # x = layers.Dense(patch_dims, activation = 'relu')(x)
    for i in range(6):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)#layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads = 4, key_dim = patch_dims_trans)(x1,x1)
        x2 = encoded_patches + attention_output 
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)#layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3,hidden_units=[patch_dims_trans, patch_dims_trans])
        encoded_patches = x2 + x3
    skeleton = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    features = mlp(skeleton ,hidden_units=[patch_dims_trans, patch_dims_trans])
    outputs = layers.Reshape((h,w, patch_dims_trans))(features) 
    
    scamp = tf.keras.layers.Conv2DTranspose(
            filters=patch_dims_trans,
            kernel_size=3,
            strides=2,
            padding="same",
        )(outputs)
    scamp = tf.keras.layers.Activation(
        activation='relu'
    )(scamp)
    scamp = tf.keras.layers.Concatenate()(
            [higher_resolutionx_8x, higher_resolutionx_4x, scamp]
        )
    scamp = tf.keras.layers.Conv2D(
        filters=patch_dims,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
    )(scamp)
    scamp = tf.keras.layers.Activation(
        activation='relu'
    )(scamp)
    scamp = tf.keras.layers.Conv2D( 
        filters=patch_dims,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
    )(scamp)
    scamp = tf.keras.layers.Activation(
        activation='relu'
    )(scamp)
    scamp = tf.keras.layers.Conv2D(
        filters=n_output_channels,
        kernel_size=1,
        strides=1,
        padding="same", name ='scamp'
    )(scamp)
    outputs = scamp
    return keras.Model(inputs,outputs, name="keypoint_detector")
IMG_SIZE_H_ori, IMG_SIZE_W_ori= 964,1288
global_scale = 0.5
IMG_SIZE_H, IMG_SIZE_W= 480,640#int(IMG_SIZE_H_ori * global_scale), int(IMG_SIZE_W_ori * global_scale)
stride = 8
NUM_KEYPOINT = 16
NUM_KEYPOINTS = NUM_KEYPOINT * 2
num_classes = 1
channels = 3
target_height,target_width = int(IMG_SIZE_H / stride), int(IMG_SIZE_W / stride)
cm = plt.get_cmap('hsv', 36)
checkpoint_path = '*/cp.ckpt'
tracker = ADPT(num_classes = 1)
tracker.load_weights(checkpoint_path)
video_paths = glob.glob('*/*.avi')
pcutoff = 0.6
kp_con = [{'name':'0_2','color':cm(27), 'bodypart':(0,1)},
        {'name':'0_1','color':cm(31),'bodypart':(0,2)},
        {'name':'2_4','color':cm(29), 'bodypart':(1,2)},
        {'name':'1_3','color':cm(33),'bodypart':(2,3)},
        {'name':'6_8','color':cm(5),'bodypart':(1,3)},
        {'name':'5_7','color':cm(10),'bodypart':(3,4)},
        {'name':'8_10','color':cm(7),'bodypart':(3,5)},
        {'name':'4_5','color':cm(7),'bodypart':(4,5)},
        {'name':'7_9','color':cm(12),'bodypart':(4,6)},
        {'name':'12_14','color':cm(16),'bodypart':(4,8)},
        {'name':'11_13','color':cm(22),'bodypart':(5,7)},
        {'name':'6_7','color':cm(22),'bodypart':(6,7)},
        {'name':'14_16','color':cm(18),'bodypart':(5,9)},
        {'name':'13_15','color':cm(24),'bodypart':(6,10)},
        {'name':'5_6','color':cm(16),'bodypart':(7,11)},
        {'name':'12_12','color':cm(16),'bodypart':(3,12)},
        {'name':'12_12','color':cm(16),'bodypart':(4,12)},
        {'name':'12_12','color':cm(16),'bodypart':(5,12)},
        {'name':'12_12','color':cm(16),'bodypart':(6,12)},
        {'name':'12_12','color':cm(16),'bodypart':(7,12)},
        {'name':'12_12','color':cm(16),'bodypart':(13,12)},
        {'name':'11_12','color':cm(22),'bodypart':(7,13)},
        {'name':'5_11','color':cm(18),'bodypart':(6,13)},
        {'name':'6_12','color':cm(24),'bodypart':(13,14)},
        {'name':'14_15','color':cm(25),'bodypart':(14,15)}]
bodyparts = [
    "nose","left_eye", "right_eye","neck","left_front_limb","right_front_limb","left_hind_limb",
    "right_hind_limb","left_front_claw","right_front_claw","left_hind_claw","right_hind_claw","back","root_tail",
    "mid_tail","tip_tail"
]
colors = [(255,182,193),(218,112,214),(75,0,130),(0,0,205),(173,216,230),(47,79,79),(240,255,240),(85,107,47),
          (253,245,230),(255,228,196),(255,160,122),(165,42,42),(105,105,105),(216,191,216),(127,255,0),(205,133,63),(178,34,34)]
step = 64
for video_path in video_paths:
    kps = []
    video = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(video_path.split('.')[0] + '_labeled.avi', fourcc, 30,(IMG_SIZE_W_ori,IMG_SIZE_H_ori)) 
    fps = 0
    if video.isOpened():
        t0 = time.time()
        rval,frame = video.read()
        f = 0
        frame_resizeds = []
        frame_resized_tensors = []
        while rval:
            fps += 1

            h,w = frame.shape[:2]

            frame_resizeds.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            rval,frame = video.read()
            f += 1
            if f % step == 0 or rval is not True:
                frame_resized_tensors = np.array(frame_resizeds)
                predictions = tracker.predict(frame_resized_tensors,verbose = 0)
                pre_len = predictions.shape[0]
                for num in range(pre_len):
                    pre = predictions[num,:,:,:]
                    lrss = np.argmax(pre[:,:,3*NUM_KEYPOINT:],2)
                    frame_resized = cv2.cvtColor(frame_resizeds[num], cv2.COLOR_BGR2RGB)
                    # frame_resized[:60,:80,0] = np.argmax(pre[:,:,3*NUM_KEYPOINT:3*NUM_KEYPOINT+2],2) * 255
                    keypoints_pre = []
                    identity = np.argmax(pre[:,:,3*NUM_KEYPOINT*num_classes:],2)
                    
                    for i in range(NUM_KEYPOINT):
                        mask_pre = pre[:, :, i]*lrss
                        a = np.unravel_index(
                                    np.argmax(mask_pre), mask_pre.shape
                                )
                        a_loc_w,a_loc_h=  pre[a[0],a[1],i+NUM_KEYPOINT],pre[a[0],a[1],i+NUM_KEYPOINT * 2]
                        keypoints_pre.append([int((a[1] - a_loc_w)*stride / IMG_SIZE_W * IMG_SIZE_W_ori), int((a[0] - a_loc_h)*stride / IMG_SIZE_H * IMG_SIZE_H_ori), pre[a[0],a[1],i]])
                        # if pre[a[0],a[1],i] < pcutoff:
                        #     keypoints_pre[-1] = None
                    kps.append(keypoints_pre)
                        # keypoints_ori.append([int(np.mean(b[0]) * 8),int(np.mean(b[1]) * 8)])
                    # print(keypoints_ori)
                    # frame = sample_val_images[img_num]
                    for i in reversed(range(len(kp_con))):
                        j1 = kp_con[i]['bodypart'][0]
                        j2 = kp_con[i]['bodypart'][1]
                        if keypoints_pre[j1][-1] > pcutoff and keypoints_pre[j2][-1] > pcutoff:
                            cv2.line(frame_resized,(keypoints_pre[j1][0], keypoints_pre[j1][1]),(keypoints_pre[j2][0], keypoints_pre[j2][1]),  color=(0,0,255), thickness = 2)
                    for idx,point in enumerate(keypoints_pre):
                        if point[-1] > pcutoff:
                            cv2.circle(frame_resized,(int(point[0]),int(point[1])),3,color = colors[idx],thickness = -1)
                    out.write(frame_resized) 
                    # cv2.imshow('monkey', frame_resized)
                    # cv2.imshow('monkey_heatmap', pre[:,:,-1])
                    # cv2.waitKey(1)
                
                f = 0
                frame_resizeds = []
                # frame_resized_tensors = []
    print(t0-time.time() )
    video.release()
    out.release()
    body_part = bodyparts
    columns = [['tang' for i in range(NUM_KEYPOINT * 3)],  [body_part[int(i/3)] for i in range(NUM_KEYPOINT * 3)]]
    xy = []
    for i in range(NUM_KEYPOINT):
        xy.append("x")
        xy.append("y")
        xy.append("likelihood")
    columns.append(xy)
    coords = np.zeros((fps, NUM_KEYPOINT * 3))
    for i,coord in enumerate(kps):
        for j,kp in enumerate(coord):
            coords[i,j*3:(j+1) * 3] = kp
    df = pd.DataFrame(columns=columns, index=[i for i in range(fps)], data=np.array(coords))
    df.columns.names =['scorer', 'bodyparts', 'coords']
    df.to_csv(video_path.split('.')[0] + '.csv')