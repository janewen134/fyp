#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET
import glob
import random

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(box,k):
    # number of boxes
    row = box.shape[0]
    
    # position of each bound
    distance = np.empty((row,k))
    
    # the final clustering position
    last_clu = np.zeros((row,))

    np.random.seed()

    # choose 5 from above
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # calculate avg for the 5 boxes
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # get min value
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # get median of each point
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    # search glob
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # get width / height of each object
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # output width / height
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


def load_my_data():
    labels = 'work_dir/train.txt'
    fr = open(labels, 'r')
    data = []
    for line in fr:
        info = line.strip().split(' ')
        for i,m in enumerate(info):
            if i == 0:
                continue
            box = m.split(',')
            xmin = np.float64(box[0])/1920
            ymin = np.float64(box[1])/1200
            xmax = np.float64(box[2])/1920
            ymax = np.float64(box[3])/1200
            
            data.append([xmax-xmin, ymax-ymin])
    return np.array(data)




if __name__ == '__main__':
    SIZE = 608
    anchors_num = 9
    data = load_my_data()
    
    # 
    out = kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
    f = open("yolo_anchors.txt", 'w')
    row = np.shape(data)[0]
    for i in range(row):
        if i == 0:
            x_y = "%d,%d" % (round(data[i][0]), round(data[i][1]))
        else:
            x_y = ", %d,%d" % (round(data[i][0]), round(data[i][1]))
        f.write(x_y)
    f.close()