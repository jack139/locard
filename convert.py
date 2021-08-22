# coding=utf-8

import os

#from model import get_model
from keras.models import load_model

chpt_file = "locard_vgg16_b4_e40_1000.hdf5"

model = load_model(chpt_file)
model.save_weights(os.path.splitext(chpt_file)[0]+'.h5')

