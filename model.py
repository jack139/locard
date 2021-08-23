# coding=utf-8

import numpy as np
from keras.applications import ResNet50, VGG16, MobileNetV2
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Flatten, LeakyReLU


def get_model(model_type='vgg16', input_size = (224,224,3)):
    # create the base pre-trained model
    if model_type=='resnet':
        base_model = ResNet50(weights='imagenet', input_shape=input_size, include_top=False)
    if model_type=='mobile':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=input_size))
    else:
        base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=input_size))

    # freeze all VGG layers so they will *not* be updated during the
    # training process
    for layer in base_model.layers:
        layer.trainable = False
    # flatten the max-pooling output of VGG
    flatten = base_model.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dense(64, activation="relu")(bboxHead)
    bboxHead = Dense(32, activation="relu")(bboxHead)

    #bboxHead = Dense(1024)(flatten)
    #bboxHead = LeakyReLU(alpha=0.02)(bboxHead)
    #bboxHead = Dropout(0.2)(bboxHead)
    #bboxHead = Dense(256)(bboxHead)
    #bboxHead = LeakyReLU(alpha=0.02)(bboxHead)
    #bboxHead = Dropout(0.2)(bboxHead)
    #bboxHead = Dense(64)(bboxHead)
    #bboxHead = LeakyReLU(alpha=0.02)(bboxHead)
    #bboxHead = Dropout(0.2)(bboxHead)

    bboxHead = Dense(8, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=base_model.input, outputs=bboxHead)

    return model
