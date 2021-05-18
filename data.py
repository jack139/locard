# coding=utf-8

import os, json
import numpy as np 
#import skimage.io as io
#import skimage.transform as trans
import cv2

def dataGenerator(data_path,json_path,target_size = (256,256)):
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
                int(j['shapes'][0]['label'])*1.
            ])

            # 准备图片
            #img = io.imread(os.path.join(data_path,i),as_gray = False)
            #img = trans.resize(img,target_size)
            img = cv2.imread(os.path.join(data_path,i))
            img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)

            #cv2.rectangle(img, (int(y[0]), int(y[1])), (int(y[2]), int(y[3])), (255,0,0), 2)
            #cv2.imwrite(os.path.join("tmp","tmp_%s"%i), img)
            #io.imsave(os.path.join("tmp","tmp_%s"%i),img.astype(np.uint8))

            img = img / 255
            img = np.reshape(img,(1,)+img.shape) # （1,256,256,3）

            y = np.reshape(y,(1,5)) # （1, 5, 1）

            # only for test
            #print(i, y)
            #print(img.shape, y.shape)

            yield (img, y)

