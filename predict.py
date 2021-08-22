# coding=utf-8

import os, sys
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np 
import cv2
from model import get_model
#from keras.models import load_model
from datetime import datetime

#model = load_model("locard_vgg16_b4_e20_1000.hdf5")

model = get_model('vgg16')
model.load_weights("locard_vgg16_b4_e40_1000.h5")

def read_img(test_path,target_size = (224,224)):
    img = cv2.imread(test_path)
    h, w = img.shape[:2]
    img = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
    img = img / 255.
    img = np.reshape(img,(1,)+img.shape)
    return img, h, w

def draw_box(test_path, p1, p2):
    img = cv2.imread(test_path)
    cv2.polylines(img, [np.array([ [p1[0], p1[1]], [p1[2], p1[1]], [p1[2], p1[3]], [p1[0], p1[3]] ], np.int32)], 
        True, color=(0, 255, 0), thickness=2)
    cv2.polylines(img, [np.array([ [p2[0], p2[1]], [p2[2], p2[1]], [p2[2], p2[3]], [p2[0], p2[3]] ], np.int32)], 
        True, color=(0, 255, 0), thickness=2)
    cv2.imwrite('data/test_result.jpg', img)


def predict(inputs, h, w): # h,w 为原始图片的 尺寸
    start_time = datetime.now()
    results = model.predict(inputs)
    print('[Time taken: {!s}]'.format(datetime.now() - start_time))

    p1 = (
        results[0][0]*w,
        results[0][1]*h,
        results[0][2]*w,
        results[0][3]*h,
    )

    p2 = (
        results[0][4]*w,
        results[0][5]*h,
        results[0][6]*w,
        results[0][7]*h,
    )

    print(results)

    return p1, p2


if __name__ == '__main__':
    if len(sys.argv)<2:
        print("usage: python %s <img_path>"%sys.argv[0])
        sys.exit(2)

    inputs, h, w = read_img(sys.argv[1])
    p1, p2 = predict(inputs, h, w)
    draw_box(sys.argv[1], p1, p2)