#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import matplotlib
# matplotlib.use('Agg')
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf 
from keras.layers import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.layers import Input,multiply,Lambda,add
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
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

    out=multiply([identity,up_h,up_w])
   
   
    return out

# model define
from keras.engine.topology import get_source_inputs
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2D, Dense, Dropout, Flatten
from keras.layers import Flatten, Dense, add, ReLU
from keras.models import Sequential
from keras.models import Model
import keras.backend as K
def AttNet(input_tensor=None, input_shape=(128,128,3),classes=7):
    if input_tensor is None:
        #print('input is none')
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            #print('convert to keras tensor')
            img_input = Input(tensor=input_tensor, shape=input_shape)
            #print('convert finished ')
        else:
            #print('not convert to keras tensor')
            img_input = input_tensor
    x=Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', 
                 input_shape = input_shape)(img_input)
    x=Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu')(x)

#     x=channel_spatial_squeeze_excite(x)
    x=MaxPooling2D(pool_size = (2, 2))(x)
    x=Dropout(0.5)(x)
    x=Conv2D(filters = 64 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x=Conv2D(filters = 128 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
#     x=rcab_block(x)
#     x=cbam(x)
#     x=channel_spatial_squeeze_excite(x)
    x=MaxPooling2D(pool_size = (2, 2))(x)
    x=Dropout(0.5)(x)
    x=Conv2D(filters = 128 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
    x=Conv2D(filters = 256 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
    
#     x=batch_aware_attention(x)
    y=rcab_block(x)
#     x=channel_spatial_squeeze_excite(x)
    x=add([x,y])
    attention=ReLU()(x)
    x=MaxPooling2D(pool_size = (2,2))(attention)
    x=Dropout(0.5)(x)
    x=Conv2D(filters = 256 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
#     x=channel_spatial_squeeze_excite(x)
    x=MaxPooling2D(pool_size = (2,2))(x)
    x=Dropout(0.5)(x)
    x=Flatten()(x)
    x=Dense(classes, activation='softmax')(x)
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = Model(inputs, x, name='batch-aware-mobilenet')
    return model  



# %%
#Helper function to load images from given directories
def load_image(src):
    image = cv2.resize(cv2.imread(src), (128, 128))
    image = image.astype('float32')/255.0
    image = tf.expand_dims(image, axis=0)
    return(image)
#%% predict
def predict(path):
    pred_x=load_image(path)
    model=AttNet(input_shape=(128,128,3),classes=7)

    # load h5 file
    model.load_weights('als_dl_model_coord_ds_us_rotate_6.h5')

    # model.summary()
    pred_y = model.predict(pred_x, steps=1)
    #print(pred_y)
    pred_y = np.argmax(pred_y, axis = 1)[0]
    uniq_labels = ['F','J','L','Q','M','nothing','space']
    result=  uniq_labels[pred_y]
    print(result)
    return pred_y
# %%
def display_attention(path):
    # pred_x=load_image(path)
    model=AttNet(input_shape=(128,128,3),classes=7)

    # load h5 file
    model.load_weights('als_dl_model_coord_ds_us_rotate_6.h5')

    # model.summary()
    # pred_y = model.predict(pred_x, steps=1)
    #print(pred_y)
    # pred_y = np.argmax(pred_y, axis = 1)[0]
    # uniq_labels = ['F','J','L','Q','M','nothing','space']
    # result=  uniq_labels[pred_y]
    #print(result)
    # return pred_y
    # african_elephant_output = model.output[:,pred_y]   # 预测向量中的非洲象元素

    # last_conv_layer = model.get_layer('re_lu')  # block5_conv3层的输出特征图，它是VGG16的最后一个卷积层

    # print(last_conv_layer.get_output_shape_at(0))
    # grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]   # 非洲象类别相对于block5_conv3输出特征图的梯度

    # pooled_grads = K.mean(grads, axis=(0, 1, 2))   # 形状是（512， ）的向量，每个元素是特定特征图通道的梯度平均大小

    # iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])  # 这个函数允许我们获取刚刚定义量的值：对于给定样本图像，pooled_grads和block5_conv3层的输出特征图
    # pooled_grads_value, conv_layer_output_value = iterate([pred_x])  # 给我们两个大象样本图像，这两个量都是Numpy数组

    # for i in range(256):
    #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i]  # 将特征图数组的每个通道乘以这个通道对大象类别重要程度

    # heatmap = np.mean(conv_layer_output_value, axis=-1)  # 得到的特征图的逐通道的平均值即为类激活的热力图

    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()




    for attention in model.layers:
        
        c_shape = attention.get_output_shape_at(0)
        # print(c_shape)
        print(attention.name, ': ', c_shape)
        # if len(c_shape)==4:
            # print(c_shape[-1])
            # if attention.name=='multiply':
            #     print(attention)
            #     break
    
# %%
if __name__=="__main__":
    # display_attention('f_04.jpg')
    result1=predict('f_04.jpg')
    # result2=predict('j_01.jpg')
    # result3=predict('j_02.jpg')
    # result4=predict('m_01.jpg')
    # result5=predict('m_02.jpg')
    # result4=predict('m_03.jpg')
    # result5=predict('m_04.jpg')
    # result6=predict('m_05.jpg')
    # result7=predict('q_02.jpg')
    # result8=predict('q_03.jpg')
    # result7=predict('l_01.jpg')
    # result8=predict('l_02.jpg')
