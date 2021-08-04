# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.applications import ResNet50, MobileNetV2, VGG16
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, Dropout, Input, Flatten
from keras.callbacks import ModelCheckpoint
from data import dataGenerator


input_size = (224,224,3)
batch_size = 4
steps_per_epoch = 1000
epochs = 20
train_dir = 'data/train_data'
train_json = 'data/json'
val_dir = 'data/val_data'
val_json = 'data/json'


# 数据生成器
train_generator = dataGenerator(train_dir, train_json, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, target_size=input_size[:2])


# create the base pre-trained model
#base_model = ResNet50(weights='imagenet', input_shape=input_size, include_top=False)
#base_model = MobileNetV2(weights=None, input_shape=input_size, include_top=False)
base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=input_size))

# freeze all VGG layers so they will *not* be updated during the
# training process
base_model.trainable = False
# flatten the max-pooling output of VGG
flatten = base_model.output
flatten = Flatten()(flatten)
# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(8, activation="sigmoid")(bboxHead)
# construct the model we will fine-tune for bounding box regression
model = Model(inputs=base_model.input, outputs=bboxHead)


# initialize the optimizer, compile the model, and show the model
# summary
#model.compile(optimizer=RMSprop(lr = 1e-4), loss='mse', metrics = ['mae'])
opt = Adam(lr=3e-4)
model.compile(loss="mse", optimizer=opt)

print(model.summary())

# train the network for bounding box regression
print("[INFO] training bounding box regressor...")

# train the model on the new data for a few epochs
#model.fit(train_generator,
#        steps_per_epoch=steps_per_epoch,
#        epochs=epochs,
#        #validation_data=val_generator,
#        #validation_steps=500
#)

model_checkpoint = ModelCheckpoint("locard_%d_%d.hdf5"%(epochs,steps_per_epoch), 
    monitor='loss',verbose=1, save_best_only=True)

model.fit_generator(train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    callbacks=[model_checkpoint]
)
