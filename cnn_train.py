# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from data import dataGenerator
from model import get_model

input_size = (224,224,3)
batch_size = 4
steps_per_epoch = 1000
epochs = 40
train_dir = 'data/train'
train_json = 'data/json'
val_dir = 'data/val'
val_json = 'data/json'


# 数据生成器
train_generator = dataGenerator(train_dir, train_json, batch_size=batch_size, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, batch_size=4, target_size=input_size[:2])

# 生成模型
model_type = 'vgg16'
model = get_model(model_type)

# initialize the optimizer, compile the model, and show the model
# summary
#model.compile(optimizer=RMSprop(lr = 1e-4), loss='mse', metrics = ['mae'])
opt = Adam(lr=3e-4)
model.compile(loss="mse", optimizer=opt)

print(model.summary())

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")


model_checkpoint = ModelCheckpoint("locard_%s_b%d_e%d_%d.hdf5"%(model_type,batch_size,epochs,steps_per_epoch), 
    monitor='val_loss',verbose=1, save_best_only=True)

model.fit_generator(train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=100,
    callbacks=[model_checkpoint]
)
