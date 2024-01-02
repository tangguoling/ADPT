# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:36:00 2022

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
from tensorflow.keras.layers import BatchNormalization

def train(model_name,save_path,evaluate = False):
    np.random.seed(train_times)
    target_height,target_width = int(IMG_SIZE_H / stride), int(IMG_SIZE_W / stride)

    with open(JSON) as infile:
        json_data = json.load(infile)
    
    json_dict = {i["img_path"]: i for i in json_data}
    
    all_data = list(json_dict.keys())
    np.random.shuffle(all_data)
    train_num = int(TrainingFraction * len(all_data))
    samples = all_data
    train_keys, validation_keys = (
        samples[:train_num],
        samples[train_num: ],
    )
    
    imgs_array = {}
    for img in samples:
        imgs_array[img] = plt.imread(os.path.join(IMG_DIR, img))
    def get_dog(name):
        data = json_dict[name]
        # img_data = plt.imread(os.path.join(IMG_DIR, data["img_path"]))
        img_data = imgs_array[data["img_path"]]
        img_data = img_as_ubyte(img_data)
        # img_data = img_data.astype(np.uint8)
        # If the image is RGBA convert it to RGB.
        if img_data.shape[-1] == 4:
            img_data = img_data.astype(np.uint8)
            img_data = Image.fromarray(img_data)
            img_data = np.array(img_data.convert("RGB"))
        data["img_data"] = img_data
    
        return data
    
    num_samples = 4
    selected_samples = np.random.choice(samples, num_samples, replace=False)
    
    images, keypoints,bbs = [], [], []
    
    for sample in selected_samples:
        data = get_dog(sample)
        image = data["img_data"]
        keypoint = data["joints"]
        bb = data["img_bbox"]
        
        images.append(image)
        keypoints.append(keypoint)
        bbs.append(bb)
    
    # In[]
    class KeyPointsDataset(keras.utils.Sequence):
        def __init__(self, image_keys, aug,heatmap_aug, batch_size=BATCH_SIZE, train=True):
            self.image_keys = image_keys
            self.aug = aug
            self.heatmap_aug = heatmap_aug
            self.batch_size = batch_size
            self.train = train
            self.on_epoch_end()
    
        def __len__(self):
            return len(self.image_keys) // self.batch_size
    
        def on_epoch_end(self):
            self.indexes = np.arange(len(self.image_keys))
            if self.train:
                np.random.shuffle(self.indexes)
    
        def __getitem__(self, index):
            indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
            image_keys_temp = [self.image_keys[k] for k in indexes]
            (images, heatmaps) = self.__data_generation(image_keys_temp)
    
            return (images, heatmaps)
    
        def __data_generation(self, image_keys_temp):
            batch_images = np.empty((self.batch_size, IMG_SIZE_H_ori, IMG_SIZE_W_ori, channels), dtype="int")
            # batch_keypoints = np.empty(
            #     (self.batch_size, 1, 1, NUM_KEYPOINTS), dtype="float32"
            # )
            batch_heatmaps = np.empty(
                (self.batch_size,target_height,target_width, NUM_KEYPOINT*3 + 1), dtype="float32"
            )
            current_images = []
            kps_objs = []
            for i, key in enumerate(image_keys_temp):
                data = get_dog(key)
                current_keypoint = np.array(data["joints"])[:, :2]
                bbx = data["img_bbox"]
                kps = []
                
                [x1,y1,x2,y2] = bbx
                x2+=x1
                y2+=y1
                # print(x1,y1,x2,y2)
                # To apply our data augmentation pipeline, we first need to
                # form Keypoint objects with the original coordinates.
                for j in range(0, len(current_keypoint)):
                    kps.append(Keypoint(x=np.nan_to_num(current_keypoint[j][0], nan = -IMG_SIZE_W), y=np.nan_to_num(current_keypoint[j][1], nan = -IMG_SIZE_H)))
    
                # We then project the original image and its keypoint coordinates.
                current_image = data["img_data"]
                # current_image *= 255
                kps_obj = KeypointsOnImage(kps, shape=current_image.shape)
                current_images.append(current_image)
                kps_objs.append(kps_obj)
            if self.aug is not None:
                (new_images, new_kps_objs) = self.aug(images=current_images, keypoints=kps_objs)
            else:
                new_images, new_kps_objs = current_images, kps_objs
                   
            (heatmap_images, heatmap_kps_objs) = self.heatmap_aug(images=new_images, keypoints=new_kps_objs)
            
            for i in range(self.batch_size):
                batch_images[i,] = new_images[i]
                heatmap_kps_obj = heatmap_kps_objs[i]
                distance_maps = heatmap_kps_obj.to_distance_maps()
                
                
                heatmaps = np.exp(-(distance_maps / variation)**2)
                height, width = target_height,target_width
                distance_maps_w = np.zeros((height, width, NUM_KEYPOINT),
                                         dtype=np.float32)
                
                distance_maps_h = np.zeros((height, width, NUM_KEYPOINT),
                                         dtype=np.float32)
                yy = np.arange(0, height)
                xx = np.arange(0, width)
                grid_xx, grid_yy = np.meshgrid(xx, yy)
                
                for num, keypoint in enumerate(heatmap_kps_obj):
                    y, x = keypoint.y, keypoint.x
                    distance_maps_w[:, :, num] = grid_xx - x
                    distance_maps_h[:, :, num] = grid_yy - y
                    distance_maps_w[:, :, num][np.where(heatmaps[:, :, num] < np.exp(-(2 / variation)**2))] = 0 
                    distance_maps_h[:, :, num][np.where(heatmaps[:, :, num] < np.exp(-(2 / variation)**2))] = 0 
    
                mask = np.zeros((height, width,1),
                                         dtype=np.float32)
                # mask[sum([heatmaps[:,:,i] for  i in range(NUM_KEYPOINT)]) >= np.exp(-variation),0] = min_distance
                kp_temp = []
                for keypoint in heatmap_kps_obj:
                    kp_temp.append([round(np.nan_to_num(keypoint.y)), round(np.nan_to_num(keypoint.x))])
                    # kp_temp.append()
                for num in reversed(range(len(kp_con))):
                    j1 = kp_con[num]['bodypart'][0]
                    j2 = kp_con[num]['bodypart'][1]
                    if kp_temp[j1][0] > 0 and kp_temp[j1][1] > 0 and kp_temp[j2][0] > 0 and kp_temp[j2][1] > 0 and kp_temp[j1][0] < height and kp_temp[j1][1] < width and kp_temp[j2][0] < height and kp_temp[j2][1] < width:
                        # print((kp_temp[j1][1], kp_temp[j1][0]),(kp_temp[j2][1], kp_temp[j2][0]))
                        cv2.line(mask,(kp_temp[j1][1], kp_temp[j1][0]),(kp_temp[j2][1], kp_temp[j2][0]),  color=1, thickness = 1)
    
                a = np.concatenate([heatmaps + np.spacing(1),distance_maps_w,distance_maps_h,mask],axis = -1)
                # print(np.max(a))
                # batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, NUM_KEYPOINTS)
                batch_heatmaps[i,] = a
    
            return (batch_images,  batch_heatmaps)
    train_aug = iaa.Sequential(
        [       
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-25, 25))),
            # iaa.Sometimes(0.33, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
            iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 1.25))),
                iaa.Sometimes(0.5, iaa.MotionBlur(k =  7, angle = [-90, 90])),
                iaa.Sometimes(0.5, iaa.CoarseDropout(0.02, size_percent=0.3, per_channel=0.5)),
                iaa.Sometimes(0.5, iaa.ElasticTransformation(sigma=5)),
                # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # iaa.Grayscale(alpha=(0.5, 1.0)),
                    # translate_percent={"x": (-0.12, 0.12), "y": (-0.20, 0.03)})),
                iaa.Sometimes(
                    0.1, iaa.AllChannelsHistogramEqualization()#"histeq"
                ),
                iaa.Sometimes(0.1, iaa.AllChannelsCLAHE()),#"clahe"
                # iaa.Sometimes(cfg_cnt["logratio"], iaa.LogContrast(**opt))#"log"
                # iaa.Sometimes(cfg_cnt["linearratio"], iaa.LinearContrast(**opt)),#"linear"
                # iaa.Sometimes(cfg_cnt["sigmoidratio"], iaa.SigmoidContrast(**opt)),#"sigmoid"
                # iaa.Sometimes(cfg_cnt["gammaratio"], iaa.GammaContrast(**opt)),#"gamma"
                # iaa.Sometimes(0.3, iaa.Sharpen(False)),#"sharpen"
                iaa.Sometimes(0.1, iaa.Emboss(alpha=(0.0,1.0),strength=(0.5,1.5))),#"emboss"
                iaa.Sometimes(0.4, iaa.CropAndPad(percent=(-0.15, 0.15))),#, keep_size=False)),#"emboss"
                # iaa.Sometimes(0.1, iaa.EdgeDetect(False)),#"edge"
                # iaa.Resize({"height": IMG_SIZE_H, "width": IMG_SIZE_W}, interpolation="linear")
        ]
    )
    
    test_aug = iaa.Sequential([iaa.Resize({"height": IMG_SIZE_H, "width": IMG_SIZE_W}, interpolation="linear")])
    
    heatmap_aug = iaa.Sequential([iaa.Resize({"height": target_height, "width": target_width}, interpolation="linear")])
    
        
    train_dataset = KeyPointsDataset(train_keys, train_aug,heatmap_aug)
    validation_dataset = KeyPointsDataset(validation_keys, None,heatmap_aug, train=False)
    print(f"Total batches in training set: {len(train_dataset)}")
    print(f"Total batches in validation set: {len(validation_dataset)}")
    # In[]
    import math
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, initial_learning_rate, decay_steps, warmup_steps,alpha=0):
        super(CustomSchedule, self).__init__()
    
        self.initial_learning_rate = initial_learning_rate
        self.initial_learning_rate = tf.cast(self.initial_learning_rate, tf.float32)
        # self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
      def __call__(self, step):
        # if step > self.warmup_steps:
        # arg1 = (self.warmup_steps ** -0.5) * self.decay_rate ^ (step / self.decay_steps)
        # else:
        arg2 = step / self.warmup_steps * self.initial_learning_rate
        
        step -= self.warmup_steps
        self.decay_steps -= self.warmup_steps
        initial_learning_rate = self.initial_learning_rate
        initial_learning_rate = tf.convert_to_tensor(
          initial_learning_rate, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        decay_steps = tf.cast(self.decay_steps, dtype)
        # decay_rate = tf.cast(self.decay_rate, dtype)
    
        global_step_recomp = tf.cast(step, dtype)
        global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
        completed_fraction = global_step_recomp / decay_steps
        cosine_decayed = 0.5 * (1.0 + tf.cos(
            tf.constant(math.pi, dtype=dtype) * completed_fraction))
    
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha
        arg1 = decayed * initial_learning_rate
    
        return tf.math.minimum(arg1, arg2)
   

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
    
    def sleap_middle(x,block_filters):
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
    
    class part_loss_adpt(keras.losses.Loss):
        def __init__(self,num_classes, alpha=0.25, gamma=2.0):
            super().__init__(
                reduction="none", name="part_loss"
            )
            self.CategoricalCrossentropy = keras.losses.CategoricalCrossentropy(from_logits=True)
            self.SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            self._num_classes = num_classes
            self._alpha = alpha
            self._gamma = gamma
        def call(self, y_true, y_pred):
            
            identity_loss = self.SparseCategoricalCrossentropy(y_true[:,:,:,-1],y_pred[:,:,:,(self._num_classes * 2 + 1)*NUM_KEYPOINT:])
            # identity_true = y_true[:,:,:,-1]
            
            part_loss = tf.square(y_true[:,:,:,:(self._num_classes * 2 + 1) * NUM_KEYPOINT] - y_pred[:,:,:,:(self._num_classes * 2 + 1)*NUM_KEYPOINT])
            mask = tf.math.logical_not(tf.math.equal(y_true[:,:,:,NUM_KEYPOINT:(self._num_classes * 2 + 1) * NUM_KEYPOINT], 0))
            mask = tf.cast(mask, dtype=part_loss.dtype)
            part_loss = 0.15 * tf.math.reduce_mean(part_loss[:,:,:,NUM_KEYPOINT:(self._num_classes * 2 + 1) * NUM_KEYPOINT] * mask) + tf.math.reduce_mean(part_loss[:,:,:,:NUM_KEYPOINT])
            total_loss = part_loss + identity_loss / NUM_KEYPOINTS / 4
            return total_loss
    checkpoint_path = 'mouse/' + model_name + '/' + save_path + "/cp.ckpt"
    
    cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                      save_best_only = True,
                                                 save_weights_only=True,
                                                 mode ='min',
                                                 verbose=0,save_freq='epoch'),
                   tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=30,
                    verbose=1,
                    restore_best_weights=True)
                   ]
    
    
    warmup_iteration = len(train_dataset) * WARMUP_EPOCHS
    iteration = len(train_dataset) * EPOCHS
    learning_rate = CustomSchedule(initial_learning_rate, iteration - warmup_iteration, warmup_iteration, alpha)
    
    if model_name == 'ADPT':
        model = ADPT(num_classes = 1)
        loss = part_loss_adpt(num_classes = 1)
        
    if evaluate:
        model.load_weights(checkpoint_path)
    if save_path == '_0':
        model.summary()
    if evaluate == False:
        model.compile(loss=loss, optimizer=tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=1e-4
            ))
        model.fit(train_dataset, validation_data = validation_dataset, epochs=EPOCHS, #verbose = 0,
              callbacks=[cp_callback])
        model.save_weights(checkpoint_path) 
    # In[]

    # if model_name=='DPK' or model_name=='SLEAP':
    tracker = model
    # else:
    #     tracker = tf.keras.models.Model(inputs=model.input,
    #                                 outputs=layers.Concatenate()([model.get_layer('scamp').output, model.get_layer('locref').output, model.get_layer('identity').output]))
    colors = [(255,182,193),(218,112,214),(75,0,130),(0,0,205),(173,216,230),(47,79,79),(240,255,240),(85,107,47),
              (253,245,230),(255,228,196),(255,160,122),(165,42,42),(105,105,105),(216,191,216),(127,255,0),(205,133,63),(178,34,34)]
    
    pcutoff = 0.2
    video_paths = ['D:/dataset/pose_data/single/rec1-K1-20220523-camera-0.avi'
                    ]
    step = 64
    kps = []
    # for video_path in video_paths:
    #     video = cv2.VideoCapture(video_path)
    #     fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    #     out = cv2.VideoWriter('mouse/' + model_name + '/' + video_path[:-4].split('/')[-1] + save_path + '_0631.avi', fourcc, 30,(IMG_SIZE_W,IMG_SIZE_H)) 
    #     fps = 0
    #     if video.isOpened():
    #         t0 = time.time()
    #         rval,frame = video.read()
    #         f = 0
    #         frame_resizeds = []
    #         frame_resized_tensors = []
    #         while rval:
    #             fps += 1
    #             if fps >= 1800 * 1:
    #                 break
    #             h,w = frame.shape[:2]
    #             # frame_resized = cv2.resize(frame,(IMG_SIZE_W,IMG_SIZE_H))
    #         #     frame_resized_tensor = tf.expand_dims(
    #         #     frame_resized, 0, name=None
    #         # )
    #             frame_resizeds.append(frame)
    #         #     frame_resized_tensors.append(frame_resized_tensor)
    #             rval,frame = video.read()
    #             f += 1
    #             if f % step == 0:
    #                 frame_resized_tensors = np.array(frame_resizeds)
    #                 predictions = tracker.predict(frame_resized_tensors,verbose = 0)
    #                 for num in range(step):
    #                     pre = predictions[num,:,:,:]
    #                     frame_resized = frame_resizeds[num]
    #                     frame_resized = cv2.resize(frame_resized,(IMG_SIZE_W,IMG_SIZE_H))
    #                     # frame_resized[:60,:80,0] = np.argmax(pre[:,:,3*NUM_KEYPOINT:3*NUM_KEYPOINT+2],2) * 255
    #                     keypoints_pre = []
    #                     # keypoints_ori = []
    #                     for i in range(NUM_KEYPOINT):
    #                         a = np.unravel_index(
    #                                     np.argmax(pre[:, :, i]), pre[:, :, i].shape
    #                                 )
    #                         a_loc_w,a_loc_h=  pre[a[0],a[1],i+NUM_KEYPOINT],pre[a[0],a[1],i+NUM_KEYPOINT * 2]
    #                         keypoints_pre.append([int((a[0] - a_loc_h) * stride),int((a[1] - a_loc_w)  * stride)])
    #                         if pre[a[0],a[1],i] < pcutoff:
    #                             keypoints_pre[-1] = None
    #                     kps.append(keypoints_pre)
    #                         # keypoints_ori.append([int(np.mean(b[0]) * 8),int(np.mean(b[1]) * 8)])
    #                     # print(keypoints_ori)
    #                     # frame = sample_val_images[img_num]
    #                     for i in reversed(range(len(kp_con))):
    #                         j1 = kp_con[i]['bodypart'][0]
    #                         j2 = kp_con[i]['bodypart'][1]
    #                         if keypoints_pre[j1] is not None and keypoints_pre[j2] is not None:
    #                             cv2.line(frame_resized,(keypoints_pre[j1][1], keypoints_pre[j1][0]),(keypoints_pre[j2][1], keypoints_pre[j2][0]),  color=(0,0,255), thickness = 1)
    #                     for idx,point in enumerate(keypoints_pre):
    #                         if point is not None:
    #                             cv2.circle(frame_resized,(int(point[1]),int(point[0])),2,color = colors[idx],thickness = -1)
    #                     out.write(frame_resized) 
    #                     # cv2.imshow('monkey', frame_resized)
    #                     # cv2.imshow('monkey_heatmap', pre[:,:,-1])
    #                     # cv2.waitKey(1)
                    
    #                 f = 0
    #                 frame_resizeds = []
    #                 # frame_resized_tensors = []
    #     print(t0-time.time() )
    #     video.release()
    #     out.release()
        # '''
     # In[]    
    def evalute(train_keys):
        rmse = [[] for i in range(NUM_KEYPOINT)]
        train_images = []
        train_keypoints =[]
        for key in train_keys:
            data = get_dog(key)
            current_img = data["img_data"]
            train_keypoints.append(np.array(data["joints"])[:, :2])
            train_images.append(current_img.astype(np.int32))
        frame_resized_tensors = np.array(train_images)
        predictions = tracker.predict(frame_resized_tensors,batch_size = BATCH_SIZE,verbose = 1)
        ap_thresholds = np.linspace(0.5, 0.95, 10)
        okss = []
        for num in range(len(train_images)):
            pre = predictions[num,]
            keypoints_pre = []
            oks = 0
            vis = 0 + np.spacing(1)
            for i in range(NUM_KEYPOINT):
                a = np.unravel_index(
                            np.argmax(pre[:, :, i]), pre[:, :, i].shape
                        )
                a_loc_w,a_loc_h=  pre[a[0],a[1],i+NUM_KEYPOINT],pre[a[0],a[1],i+NUM_KEYPOINT * 2]
                keypoints_pre.append([(a[0] - a_loc_h) * stride / IMG_SIZE_H * IMG_SIZE_H_ori,
                                      (a[1] - a_loc_w)  * stride / IMG_SIZE_W * IMG_SIZE_W_ori])
                if pre[a[0],a[1],i] <= pcutoff:
                    keypoints_pre[-1] = None
                    
            minx = min([train_keypoints[num][i][0] for i in range(NUM_KEYPOINT)])
            maxx = max([train_keypoints[num][i][0] for i in range(NUM_KEYPOINT)])
            miny = min([train_keypoints[num][i][1] for i in range(NUM_KEYPOINT)])
            maxy = max([train_keypoints[num][i][1] for i in range(NUM_KEYPOINT)])
            area = (maxx - minx) * (maxy - miny) + np.spacing(1)#* IMG_SIZE_H / IMG_SIZE_H_ori * IMG_SIZE_W / IMG_SIZE_W_ori 
            
            for i in range(NUM_KEYPOINT): 
                if keypoints_pre[i] is not None:
                    err = (train_keypoints[num][i][0] - keypoints_pre[i][1]) ** 2 + (train_keypoints[num][i][1] - keypoints_pre[i][0]) ** 2
                    rmse[i].append(math.sqrt(err) / 2)
                    oks += np.exp(-err / (2 * area * (2 * delta)**2))
                    vis += 1
            oks /= vis
            okss.append(oks)
        rmses = []
        for _ in rmse:
            rmses.append(np.nanmean(_))
        
        AP = []
        for ap_threshold in ap_thresholds:
            cnt = 0
            for oks in okss:
                if oks >= ap_threshold:
                    cnt += 1
            AP.append(cnt/len(okss))
        return rmses, AP
    pcutoff = 0.2
    rmses_train,AP_train = evalute(train_keys)
    # print(rmses_train)
    rmses_eval,AP_eval = evalute(validation_keys)
    
    pcutoff = 0.6
    rmses_train_06,AP_train_06 = evalute(train_keys)
    # print(rmses_train)
    rmses_eval_06,AP_eval_06 = evalute(validation_keys)
    
    rmses_train = rmses_train + rmses_train_06
    AP_train = AP_train + AP_train_06
    
    rmses_eval = rmses_eval + rmses_eval_06
    AP_eval = AP_eval + AP_eval_06
    # rmses_eval,AP_eval = evalute(validation_keys)
    # print(rmses_eval)
    return rmses_train,rmses_eval,AP_train,AP_eval,kps

    # In[]
model_names = ['ADPT']
IMG_SIZE_H_ori, IMG_SIZE_W_ori = 964,1288
global_scale = 0.5
IMG_SIZE_H, IMG_SIZE_W= 480,640#int(IMG_SIZE_H_ori * global_scale), int(IMG_SIZE_W_ori * global_scale)
num_classes = 1
BATCH_SIZE = 8
variation = 17 / 8
delta = 0.025
initial_learning_rate = 1e-3
alpha = 1e-5
EPOCHS = 200
WARMUP_EPOCHS = 10
NUM_KEYPOINT = 16
NUM_KEYPOINTS = NUM_KEYPOINT * 2
save_freq = 500
train_num = 10000
stride = 8
TrainingFraction = 0.95
Tranfer_LR = 1e-3
channels = 3
# Tranfer_EPOCH = 50
IMG_DIR = "D:/ADPT/data/mouse/labeled-data"
JSON = 'D:/ADPT/data/mouse.json'
imgs = glob.glob("D:/ADPT/data/mouse/labeled-data/*/*.png")

cm = plt.get_cmap('hsv', 36)
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
    
train_rmses = []
test_rmses = []
train_APs = []
test_APs = []
initial_weight = None
for model_name in model_names:
    model_rmse = []
    for train_times in range(5,10):
        evaluate = True # False
        save_path = '_' + str(train_times)
        model_rmse.append(train(model_name, save_path,evaluate))
        a = np.array(model_rmse[-1][:2], dtype='float16')
        np.savetxt('mouse/' + model_name + '/' + save_path + '_rmse.csv',a, delimiter=",")
        b = np.array(model_rmse[-1][2:-1], dtype='float16')
        np.savetxt('mouse/' + model_name + '/' + save_path + '_ap.csv',b, delimiter=",")
    train_rmse = []
    test_rmse = []
    train_AP = []
    test_AP = []
    for _ in model_rmse:
        train_rmse.append(np.mean(_[0][:NUM_KEYPOINT]))
        test_rmse.append(np.mean(_[1][:NUM_KEYPOINT]))
        train_AP.append(np.mean(_[2][:10]))
        test_AP.append(np.mean(_[3][:10]))
    train_rmses.append(train_rmse)
    
    test_rmses.append(test_rmse)
    train_APs.append(train_AP)
    test_APs.append(test_AP)

print(train_rmse)
print(test_rmse)
print(test_AP)
