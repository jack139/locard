# coding=utf-8

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
from keras.applications import ResNet50, MobileNetV2
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from data import dataGenerator


input_size = (224,224,3)
batch_size = 4
steps_per_epoch = 1000
epochs = 20
train_dir = 'data/test_data'
train_json = 'data/test_json'
val_dir = 'data/val_data'
val_json = 'data/val_json'


# 数据生成器
train_generator = dataGenerator(train_dir, train_json, target_size=input_size[:2])
val_generator = dataGenerator(val_dir, val_json, target_size=input_size[:2])


# create the base pre-trained model
#base_model = ResNet50(weights='imagenet', input_shape=input_size, include_top=False)
base_model = MobileNetV2(weights=None, input_shape=input_size, include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5)(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional ResNet50 layers
#for layer in base_model.layers:
#    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=RMSprop(lr = 1e-4), loss='mse', metrics = ['mae'])

model.summary()

# train the model on the new data for a few epochs
#model.fit(train_generator,
#        steps_per_epoch=steps_per_epoch,
#        epochs=epochs,
#        #validation_data=val_generator,
#        #validation_steps=500
#)

model.fit_generator(train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    #callbacks=[model_checkpoint]
)

#model.save('batch_%d_epochs_%d_steps_%d_0.hdf5'%(batch_size, epochs, steps_per_epoch))

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

'''
# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 100 layers and unfreeze the rest:
for layer in model.layers[:100]:
   layer.trainable = False
for layer in model.layers[100:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

#model.summary()

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit(train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=test_generator,
        validation_steps=800)

model.save('batch_%d_epochs_%d_steps_%d_1.hdf5'%(batch_size, epochs, steps_per_epoch))
'''