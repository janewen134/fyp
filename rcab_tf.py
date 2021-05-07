import matplotlib
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf 
from keras.layers import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.layers import Input,multiply,Lambda,add
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Dense, Dropout, Flatten,add, ReLU
from keras.layers import Dense
from keras.models import Sequential
from keras.models import Model
import keras.backend as K
from keras.layers import Flatten, Dense
from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# helper pooling function
def adaptivepooling(x,outsize_h,outsize_w,type='avg'):
    x_shape=K.int_shape(x)
    batchsize1,dim1,dim2,channels1=x_shape
    stride_h=np.floor(dim1/outsize_h).astype(np.int32)
    kernel_h=dim1-(outsize_h-1)*stride_h
    stride_w=np.floor(dim2/outsize_w).astype(np.int32)
    kernel_w=dim2-(outsize_w-1)*stride_w
    if type=='max':
        adpooling=MaxPooling2D(pool_size=(kernel_h,kernel_w),strides=(stride_h,stride_w))(x)
    elif type=='avg':
        adpooling=AveragePooling2D(pool_size=(kernel_h,kernel_w),strides=(stride_h,stride_w))(x)
    return adpooling

# attention block
def rcab_block(x=None, reduction=16):
    identity=x

    n,h,w,c=K.int_shape(x)
    #  1D Global Average Pooling
    x_h=adaptivepooling(x,h,1)
    x_h = Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(x_h) # for channels
    x_w=adaptivepooling(x,1,w)
    
    x_h=BatchNormalization()(x_h)
    x_w=BatchNormalization()(x_w)

    # downsample->upsample
    mip=c//reduction
    ds_h=Conv2D(filters=mip,kernel_size=1,activation='relu')(x_h)
    ds_w=Conv2D(filters=mip,kernel_size=1,activation='relu')(x_w)
    
    up_h=Conv2D(filters=c,kernel_size=1,activation='sigmoid')(ds_h)
    up_w=Conv2D(filters=c,kernel_size=1,activation='sigmoid')(ds_w)

    y=multiply([identity,up_h,up_w])
    x=add([x,y])
    out = ReLU()(x)
    return out


