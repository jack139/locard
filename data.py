# coding=utf-8

import os, json
import numpy as np 
import cv2

def dataGenerator(data_path,json_path,target_size = (224,224)):
    while True:
        file_list = os.listdir(data_path)
        file_list = sorted(file_list)
        for i in file_list:
            # 准备标记
            json_file = os.path.join(json_path, os.path.splitext(i)[0]+'.json')
            with open(json_file) as fp:
                j = json.load(fp)
            ratio_x = target_size[0] / j['imageWidth']
            ratio_y = target_size[1] / j['imageHeight']
            y = np.array([
                j['shapes'][0]['points'][0][0]*ratio_x,
                j['shapes'][0]['points'][0][1]*ratio_y,
                j['shapes'][0]['points'][1][0]*ratio_x,
                j['shapes'][0]['points'][1][1]*ratio_y,
                j['shapes'][1]['points'][0][0]*ratio_x,
                j['shapes'][1]['points'][0][1]*ratio_y,
                j['shapes'][1]['points'][1][0]*ratio_x,
                j['shapes'][1]['points'][1][1]*ratio_y,                
            ])

            # 准备图片
            img = cv2.imread(os.path.join(data_path,i))
            img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)

            #cv2.rectangle(img, (int(y[0]), int(y[1])), (int(y[2]), int(y[3])), (255,0,0), 2)
            #cv2.rectangle(img, (int(y[4]), int(y[5])), (int(y[6]), int(y[7])), (255,0,0), 2)
            #cv2.imwrite(os.path.join("tmp","tmp_%s"%i), img)

            img = img / 255.
            img = np.reshape(img,(1,)+img.shape) # （1,256,256,3）

            y = np.reshape(y,(1,8)) # （1, 8, 1）

            # only for test
            #print(i, y)
            #print(img.shape, y.shape)

            yield (img, y)
