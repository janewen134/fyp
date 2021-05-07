#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------#
#       YOLO: no attention version
# -------------------------------------#
import os
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages') # append back in order to import rospyimport numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from noattneck import YoloBody
from utils.utils import *
from yolo_layer import *
import sys
from sensor_msgs.msg import Image
import matplotlib.pyplot as plt


import rospy
from cv_bridge import CvBridge


class Inference(object):
    def __init__(self, **kwargs):
        # ros settings
        image_topic = rospy.get_param("~image_topic","")
        self._cv_bridge = CvBridge()
        # self._pub = rospy.Publisher('yolo_result', Int16, queue_size=1)

        # yolo model settings
        self.model_path = kwargs['model_path']
        self.anchors_path = kwargs['anchors_path']
        self.classes_path = kwargs['classes_path']
        self.model_image_size = kwargs['model_image_size']
        self.confidence = kwargs['confidence']
        self.cuda = kwargs['cuda']

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()
        print(self.anchors)
        self.net = YoloBody(3, len(self.class_names)).eval()
        self.load_model_pth(self.net, self.model_path)

        if self.cuda:
            self.net = self.net.cuda()
            self.net.eval()

        print('Finished!')

        self.yolo_decodes = []
        anchor_masks = [[0,1,2],[3,4,5],[6,7,8]]
        for i in range(3):
            head = YoloLayer(self.model_image_size, anchor_masks, len(self.class_names),
                                               self.anchors, len(self.anchors)//2).eval()
            self.yolo_decodes.append(head)


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)

    def callback(self,image_msg):
        ##############################################
        # convert format
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")
        # frame=np.uint8(cv_image)
        boxes = model.detect_image(cv_image)
        img=plot_boxes_cv2(cv_image, boxes,  class_names=self.class_names)
        plt.ion()
        plt.figure(1)
        plt.imshow(img)
        plt.axis('off')
        plt.pause(0.5)
        plt.ioff()
        # img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # cv2.imshow('YOLO',img)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()


    def main(self):
        rospy.spin()

    def load_model_pth(self, model, pth):
        print('Loading weights into state dict, name: %s' % (pth))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pth, map_location=device)
        matched_dict = {}

        for k, v in pretrained_dict.items():
            if np.shape(model_dict[k]) == np.shape(v):
                matched_dict[k] = v
            else:
                print('un matched layers: %s' % k)
        print(len(model_dict.keys()), len(pretrained_dict.keys()))
        print('%d layers matched,  %d layers miss' % (
        len(matched_dict.keys()), len(model_dict) - len(matched_dict.keys())))
        model_dict.update(matched_dict)
        model.load_state_dict(pretrained_dict)
        print('Finished!')
        return model

    # ---------------------------------------------------#
    #   get class names
    # ---------------------------------------------------#
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   get bounding boxes 
    # ---------------------------------------------------#
    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return anchors
        #return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


    # ---------------------------------------------------#
    #    making detection
    # ---------------------------------------------------#
    def detect_image(self, image_src):
        h, w, _ = image_src.shape
        image = cv2.resize(image_src, (416, 416))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.array(image, dtype=np.float32)
        img = np.transpose(img / 255.0, (2, 0, 1))
        images = np.asarray([img])

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        print(output.shape)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=0.1)
        # print(batch_detections) 
        # nothing is detected  
        if batch_detections == [None]:  
            return None                                
        boxes = [box.cpu().numpy() for box in batch_detections]
        print(boxes[0])
        return boxes[0]

if __name__ == '__main__':
    params = {
        "model_path": '/home/jing/ros_project/src/robot_learning/object_detect/scripts/pth/yolo4_weights.pth',
        "anchors_path": '/home/jing/ros_project/src/robot_learning/object_detect/scripts/work_dir/yolo_anchors_coco.txt',
        "classes_path": '/home/jing/ros_project/src/robot_learning/object_detect/scripts/work_dir/coco_classes.txt',
        "model_image_size": (416, 416, 3),
        "confidence": 0.4,
        "cuda": False
    }

    global model
    model = Inference(**params)

    rospy.init_node('object_detect_node')
    rospy.loginfo("yolo has started.")

    model.main()

    

    