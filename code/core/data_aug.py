# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 03:29:02 2024

@author: tang
"""

import imgaug.augmenters as iaa
def data_augmentation():
    aug = iaa.Sequential(
        [       
            iaa.Sometimes(0.5, iaa.Affine(rotate=(-25, 25))),
            iaa.Sometimes(0.33, iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})),
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
    return aug