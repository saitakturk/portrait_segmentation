import argparse
from modely import *
from keras.models import load_model
import random
import os
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD,Adam, RMSprop
import numpy as np 
import matplotlib.pyplot as plt 



fname = "model_weights-{val_acc:.4f}-{val_loss:.4f}.hdf5"

#checkpoint creation
checkpoint = ModelCheckpoint(fname, monitor="val_loss",
save_best_only=True, verbose=1)
callbacks = [checkpoint]

#>>>::: Data Augmentation custom functions
def brightness_adjustment(img):
    #print(img.shape)
    if(img.shape[2] == 3):
      # turn the image into the HSV space
      hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
      # creates a random bright
      ratio = .5 + np.random.uniform(-0.499,0.799)
      # convert to int32, so you don't get uint8 overflow
      # multiply the HSV Value channel by the ratio
      # clips the result between 0 and 255
      # convert again to uint8
      hsv[:,:,2] =  np.clip(hsv[:,:,2].astype(np.int32) * ratio, 0, 255).astype(np.uint8)
      # return the image int the BGR color space
      return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)
    
    return img
  
def img_compression(img):
    if img.shape[2] == 3:
    
      encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50 + np.random.uniform(-40,20)]
      result, encimg = cv2.imencode('.jpg', img, encode_param)
      img = cv2.imdecode(encimg, 1)
      
      return brightness_adjustment( img).astype(np.float64)
    return img

#>>>::: Keras Image Data Generator 
data_gen_args = dict(preprocessing_function= img_compression,rescale = 1/255.0)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed= 1

image_generator = image_datagen.flow_from_directory(
    'train',
    class_mode=None,
    target_size = (size,size),
    classes = ['data'],
    color_mode ='rgb',
    batch_size = batch,
    shuffle=True,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'train',
    class_mode=None,
    classes = ['label_1'],
    target_size = (size,size),
    color_mode = 'grayscale',
    batch_size = batch,
    
    seed=seed)

train_generator = zip(image_generator, mask_generator)

vimage_generator = image_datagen.flow_from_directory(
    'val',
    class_mode=None,
    classes = ['data'],
    target_size = (size,size),
    color_mode ='rgb',
    batch_size = batch,
    shuffle = True,
    seed=seed)

vmask_generator = mask_datagen.flow_from_directory(
    'val',
    class_mode=None,
    classes = ['label_1'],
    target_size = (size,size),
    color_mode = 'grayscale',
    batch_size = batch, 
    seed=seed)
val_generator = zip(vimage_generator, vmask_generator)

#>>>::: Optimizer and Metric
model = unet(None,input_size = (size,size,3))#load_model("human_weights-004-0.1119.hdf5", custom_objects={"awesomeq_loss":awesomeq_loss})#
model.compile(optimizer = Adam(lr = 1e-4), loss = awesomeq_loss, metrics = ['accuracy'])
H = model.fit_generator(
        train_generator,
   callbacks = callbacks,
        steps_per_epoch=6985//batch,
        epochs=20,
        verbose = 1,
        validation_data=val_generator,
        validation_steps=500//batch)

        
​plt.style.use("ggplot")
fig = plt.figure()


plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
title = "0.01"
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")

plt.ylabel("Loss/Accuracy")
plt.legend()
fig.savefig('image.png', dpi=fig.dpi)
plt.show()

