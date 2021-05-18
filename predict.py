# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np 
import skimage.io as io
import skimage.transform as trans
from keras.models import load_model

def testGenerator(test_path,target_size = (256,256),as_gray = False):
    file_list = os.listdir(test_path)
    file_list = sorted(file_list)
    for i in file_list:
        img = io.imread(os.path.join(test_path,i),as_gray = as_gray)
        #img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if as_gray else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

if __name__ == '__main__':
    model = load_model("batch_32_epochs_1_steps_2000_1.hdf5")
    test_path = "../datagen/data/point/test/1"
    testGene = testGenerator(test_path, target_size=(224,224))
    file_list = os.listdir(test_path)
    #results = model.predict_generator(testGene,len(file_list),verbose=1)
    inputs = next(testGene)
    results = model.predict(inputs)
    print(results)