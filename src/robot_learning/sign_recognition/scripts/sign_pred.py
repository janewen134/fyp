#!/usr/bin/env python 
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import os
import tensorflow as tf 
from keras.layers import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.layers import Input,multiply,Lambda,add
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Flatten, Dense, add, ReLU
from keras.models import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.models import load_model
# from PIL import Image # avoid conflict with message of ssensor_msgs
import PIL.Image as pilim
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from mrobot_navigation.srv import Sbs, SbsResponse
import random
# video_capture = cv2.VideoCapture(1)

class ALS_Predict():
    global graph
    graph = tf.get_default_graph()
    def __init__(self):
        # prediction model
        self._model = self.BatchAttNet(input_shape=(128,128,3),classes=7)
        self._model.load_weights('/home/jing/ros_project/src/robot_learning/sign_recognition/model/als_dl_model_coord_ds_us_rotate_6.h5')
        # self._model.summary()
        # ros
        image_topic = rospy.get_param("~image_topic", "")
        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber(image_topic, Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('als_result', Int16, queue_size=1)

    def callback(self, image_msg):
        # convert format
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        frame=np.uint8(cv_image)/255.
        
        plt.ion()
        plt.figure(2)
        plt.imshow(cv_image)
        plt.axis('off')
        plt.pause(0.5)
        plt.ioff()
        
        # #print cv_image
        global result
        result = self.predict(frame)
        
        # publish result
        uniq_labels = ['F','J','L','Q','M','nothing','space']
        result_label=  uniq_labels[result]
        rospy.loginfo('result: %s' % result_label)
        self._pub.publish(result)
        # rospy.sleep() 

    # helper pooling function
    def adaptivepooling(self,x,outsize_h,outsize_w,type='avg'):
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
    def ca_block(self,x=None, reduction=16):
        identity=x
        #print('input shape: ',x.shape)
        n,h,w,c=K.int_shape(x)
        x_h=self.adaptivepooling(x,h,1)
        x_h = Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(x_h) # for channels
        x_w=self.adaptivepooling(x,1,w)
        
        x_h=BatchNormalization()(x_h)
        x_w=BatchNormalization()(x_w)
        #print('x_h shape: ',x_h.shape)
        #print('x_w shape: ',x_w.shape)
        # downsample->upsample
        mip=c//reduction
        ds_h=Conv2D(filters=mip,kernel_size=1,activation='relu')(x_h)
        ds_w=Conv2D(filters=mip,kernel_size=1,activation='relu')(x_w)
        
        up_h=Conv2D(filters=c,kernel_size=1,activation='sigmoid')(ds_h)
        up_w=Conv2D(filters=c,kernel_size=1,activation='sigmoid')(ds_w)

        out=multiply([identity,up_h,up_w])
        #print('out shape: ',out.shape)
        return out

    # model define

    def BatchAttNet(self,input_tensor=None, input_shape=(128,128,3),classes=7):
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

        # img_input = Input(shape=input_shape)

        x=Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu', 
                    input_shape = input_shape)(img_input)
        x=Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu')(x)
        x=MaxPooling2D(pool_size = (2, 2))(x)
        x=Dropout(0.5)(x)
        x=Conv2D(filters = 64 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
        x=Conv2D(filters = 128 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
        x=MaxPooling2D(pool_size = (2, 2))(x)
        x=Dropout(0.5)(x)
        x=Conv2D(filters = 128 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
        x=Conv2D(filters = 256 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
        y=self.ca_block(x)
        x=add([x,y])
        x=ReLU()(x)
        x=MaxPooling2D(pool_size = (2,2))(x)
        x=Dropout(0.5)(x)
        x=Conv2D(filters = 256 , kernel_size = 3, padding = 'same', activation = 'relu')(x)
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
        model = Model(inputs, x, name='coord-attention-mobilenet')
        return model  

    # %%
    #Helper function to load images from given directories
    def load_image(self,src):
        # image = cv2.resize(cv2.imread(src), (128, 128))
        image = cv2.resize(src, (128, 128))
        image = image.astype('float32')/255.0
        image = tf.expand_dims(image, axis=0)
        return(image)

    # %% predict
    def predict(self,src):
        with graph.as_default():
            
            pred_x=self.load_image(src)
            pred_y = self._model.predict(pred_x, steps=1)
            pred_y = np.argmax(pred_y, axis = 1)[0]

            return pred_y

    def main(self):
        rospy.spin()

def sbsCallback(req):
    return SbsResponse(result)
# %%
if __name__=="__main__":
    # init the als prediction node
    rospy.init_node('als_prediction_node')
    # create the server for sign recognition
    # init an instance of als prediction
    als_result = ALS_Predict()
    rospy.loginfo("als_prediction_node has started.")
    service = rospy.Service('/get_sign', Sbs, sbsCallback)

    # call the callback function for the service 
    als_result.main()
    # result=predict('test34.jpg')