import numpy as np 
import os
import cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf 
#from keras import backend as keras

def unet(pretrained_weights = None,input_size = (224,224,3)):
    if input_size[0] == 224 :
        print("pretrained")
        model = MobileNetV2(include_top = False, weights = 'imagenet', input_shape = (224,224,3))
        for layer in model.layers:
            layer.trainable = False
    else:
        print("not pretrained")
        model = MobileNetV2( include_top = False,weights = None, input_shape  = input_size )
    
    conv1 = Conv2DTranspose(320, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(model.layers[-1].output)
    con1 = concatenate([model.get_layer('block_16_project_BN').output, conv1], axis=3)
    conv2 = Conv2DTranspose(320, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(con1)
    conv3 = Conv2DTranspose(192, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv2)

    up1 = UpSampling2D(size = (2,2))(conv3)
    conv4 = Conv2DTranspose(192, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(up1)
    con2 = concatenate([model.get_layer("block_12_add").output, model.get_layer("block_11_add").output, conv4], axis=3)
    conv5 = Conv2DTranspose(192, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(con2)
    conv6 = Conv2DTranspose(64, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv5)

    # conv_b_1_1 = Conv2DTranspose(16, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv6)
    # conv_b_1_2 = Conv2DTranspose(1, 3, activation = 'relu', padding='same', kernel_initializer='he_normal', name='output5')(conv_b_1_1)

    up2 = UpSampling2D(size = (2,2))(conv6)
    conv7 = Conv2DTranspose(64, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(up2)
    con3 = concatenate([model.get_layer("block_5_add").output, model.get_layer("block_4_add").output, conv7], axis=3)
    conv8 = Conv2DTranspose(64, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(con3)
    conv9 = Conv2DTranspose(32, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv8)

    # conv_b_2_1 = Conv2DTranspose(8, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv9)
    # conv_b_2_2 = Conv2DTranspose(1, 3, activation = 'relu', padding='same', kernel_initializer='he_normal', name='output4')(conv_b_2_1)
    

    up3 = UpSampling2D(size = (2,2))(conv9)
    conv10 = Conv2DTranspose(24, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(up3)
    con4 = concatenate([model.get_layer("block_2_add").output,  conv10], axis=3)
    conv11 = Conv2DTranspose(24, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(con4)
    conv12 = Conv2DTranspose(16, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv11)

    #conv_b_3_1 = Conv2DTranspose(4, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv12)
    #conv_b_3_2 = Conv2DTranspose(1, 3, activation = 'relu', padding='same', kernel_initializer='he_normal', name='output3')(conv_b_3_1)

    up4 = UpSampling2D(size = (2,2))(conv12)
    conv13 = Conv2DTranspose(16, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(up4)
    con5 = concatenate([model.get_layer("expanded_conv_project_BN").output,  conv13], axis=3)
    conv14 = Conv2DTranspose(16, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(con5)
    conv15 = Conv2DTranspose(8, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv14)

    # conv_b_4_1 = Conv2DTranspose(2, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv15)
    # conv_b_4_2 = Conv2DTranspose(1, 3, activation = 'relu', padding='same', kernel_initializer='he_normal', name='output2')(conv_b_4_1)

    up5 = UpSampling2D(size = (2,2))(conv15)
    conv16 = Conv2DTranspose(4, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(up5)
    conv17 = Conv2DTranspose(2, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv16)
    conv18 = Conv2D(1, 1, activation = 'sigmoid', name ='output1')(conv17)
    #conv4 = Conv2DTranspose(128, 3, activation = 'relu', padding='same', kernel_initializer='he_normal')(conv3)



    model2 = Model(input = model.layers[0].input, output = conv18 )

    #model2.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(model2.summary())
    return model2




def padding(img, dimension= 224, channel = 3, show_pad = False ):
    '''
    This function creates zeros padding for non-square colored images 
    '''
    empty = np.zeros((dimension, dimension,channel)).astype(np.int)
    width = img.shape[0]
    height = img.shape[1]
    
    diff_width  = (dimension - width)//2
    diff_height = (dimension - height)//2
    signal = 0
    
    if ( width == dimension and height != dimension):
        empty[:, diff_height:(height+diff_height),:] = img.astype(np.int)
        signal = 2
    elif ( width != dimension and height == dimension):
        empty[ diff_width:(width+diff_width), : ,:] = img.astype(np.int)
        signal = 1
    elif ( width <= dimension and height <= dimension ):
        empty[diff_width:(diff_width+width), diff_height:(diff_height+height),:] = img.astype(np.int)
        signal = 0
    if show_pad:
        if signal == 1:
            return empty, diff_width, signal
        elif signal == 2:
            return empty, diff_height, signal
        return empty,-1,-1
    else:
        return empty
def resize_image(img, reshape_size= 224):
    '''
    This function resizes image with saving aspect ratio
    '''
    max_shape = np.max(img.shape)
    ratio = (max_shape / reshape_size).astype(np.float64)

    new_width = np.round((img.shape[1] / ratio)).astype(np.int)
    new_height  = np.round((img.shape[0] / ratio)).astype(np.int)
   
    return cv2.resize(img, (new_width, new_height))


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def awesomeq_loss(y_true, y_pred):
    binary = K.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return binary + dice

def awesome_loss( y_true, y_pred):
    dice_lossx = dice_loss(y_true, y_pred)
    Mean_IOUx = iou_loss_core(tf.identity(y_true), tf.identity(y_pred))
    alpha = 0.7
    return  tf.to_float(alpha * Mean_IOUx + (1-alpha) * dice_lossx)


# def foreground_sparse_accuracy(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     y_pred = K.reshape(y_pred, (-1, nb_classes))
#     y_true = K.reshape(y_true, (-1, nb_classes))
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     true_pixels = K.argmax(y_true, axis=-1)
#     unpacked = tf.unstack(y_true, axis=-1)
#     legal_labels = tf.cast(unpacked[0], tf.bool) | ~K.greater(K.sum(y_true, axis=-1), 0)
#     return K.sum(tf.to_float(~legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(~legal_labels))

# def background_sparse_accuracy(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     y_pred = K.reshape(y_pred, (-1, nb_classes))
#     y_true = K.reshape(y_true, (-1, nb_classes))
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     true_pixels = K.argmax(y_true, axis=-1) 
#     legal_labels = K.greater(K.sum(y_true, axis=-1), 0) & K.equal(true_pixels, 0)
#     return K.sum(tf.to_float(legal_labels & K.equal(true_pixels, pred_pixels))) / K.sum(tf.to_float(legal_labels))

# def sparse_accuracy_ignoring_last_label(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     y_pred = K.reshape(y_pred, (-1, nb_classes))
#     y_true = K.reshape(y_true, (-1, nb_classes))
#     legal_labels = K.greater(K.sum(y_true, axis=-1), 0)
#     return K.sum(tf.to_float(legal_labels & K.equal(K.argmax(y_true, axis=-1),
#                                                     K.argmax(y_pred, axis=-1)))) / K.sum(tf.to_float(legal_labels))

# def sparse_crossentropy_ignoring_last_label(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     y_true = K.one_hot(tf.to_int32(y_true[:,:,0]), nb_classes+1)[:,:,:-1]
#     return K.categorical_crossentropy(y_true, y_pred)

# def sparse_Mean_IOU(y_true, y_pred):
#     nb_classes = K.int_shape(y_pred)[-1]
#     iou = []
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     for i in range(0, nb_classes): # exclude first label (background) and last label (void)
#         true_labels = K.equal(y_true[:,:,0], i)
#         pred_labels = K.equal(pred_pixels, i)
#         inter = tf.to_int32(true_labels & pred_labels)
#         union = tf.to_int32(true_labels | pred_labels)
#         legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
#         ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
#         iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
#     iou = tf.stack(iou)
#     legal_labels = ~tf.debugging.is_nan(iou)
#     iou = tf.gather(iou, indices=tf.where(legal_labels))
#     return K.mean(iou)

# def Mean_IOU(y_true, y_pred):
#     y_true = tf.cast(y_true, tf.int64)
#     nb_classes = K.int_shape(y_pred)[-1]
#     iou = []
#     true_pixels = K.argmax(y_true, axis=-1)
#     pred_pixels = K.argmax(y_pred, axis=-1)
#     void_labels = K.equal(K.sum(y_true, axis=-1), 0)
#     for i in range(0, nb_classes): # exclude first label (background) and last label (void)
#         true_labels = K.equal(true_pixels, i) & ~void_labels
#         pred_labels = K.equal(pred_pixels, i) & ~void_labels
#         inter = tf.to_int64(true_labels & pred_labels)
#         union = tf.to_int64(true_labels | pred_labels)
#         legal_batches = K.sum(tf.to_int64(true_labels), axis=1)>0
#         ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
#         iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
#     iou = tf.stack(iou)
#     legal_labels = ~tf.debugging.is_nan(iou)
#     iou = tf.gather(iou, indices=tf.where(legal_labels))
#     return K.mean(iou)

