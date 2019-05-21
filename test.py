import argparse
from modely import *
import cv2
import numpt as np
import matplotlib.pyplot as plt
from keras.models import load_model

def load_image(path, size=512, isImg = False):
    if isImg:
        imgx = path
    else:
        imgx = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.uint8)

    img  = imgx.copy().astype(np.uint8)
    imgx,pad,signal = padding(resize_image(imgx, size),size,3, True)
    img  = padding(resize_image(img, size),size,3).astype(np.float64)
    img /= 255.0
    img = np.expand_dims(img, axis =0)
    return imgx,img, pad,signal

def fill_holes(im_in):
    # Read image

    # Threshold.
    # Set values equal to or above 220 to 0.
    # Set values below 220 to 255.

    th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th.astype(np.int) | im_floodfill_inv.astype(np.int)

    return im_out


def potrait_extract(model,path, isImg = False,isRGB= False):
    if isImg == False:
        imgx = cv2.imread(path)#,cv2.COLOR_BGR2RGB)
    else:
        imgx = path
    img,img_pre,pad,signal = load_image(path,512, isImg)
    
    pred = model.predict(img_pre)
    mask = np.squeeze(pred) * 255.0
    mask[ mask >128] = 255
    mask[ mask <= 128 ] = 0
    if signal == 2:
        new_mask = cv2.resize(mask[:,pad:-pad], (imgx.shape[1], imgx.shape[0]),interpolation = cv2.INTER_CUBIC)
    elif signal == 1:
        new_mask = cv2.resize(mask[pad:-pad,:], (imgx.shape[1], imgx.shape[0]),interpolation = cv2.INTER_CUBIC)
    else:
        new_mask = cv2.resize(mask, (imgx.shape[1], imgx.shape[0]),interpolation = cv2.INTER_CUBIC)
    new_mask[ new_mask > 128] = 255

    new_mask[ new_mask <= 128 ] = 0
    ###computer vision - postprocess of the mask
#     kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
#     new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
    #     new_mask  = fill_holes(new_mask)
    ############################################
    imgx[ new_mask != 255 ] = 255
    if  isRGB:
        return cv2.cvtColor(imgx,cv2.COLOR_BGR2RGB)
    else:
        return imgx
parser = argparse.ArgumentParser()

parser.add_argument('-i', '--img_path', help='Input Image Path', default='img.png',  required=True)
parser.add_argument('-m','--model', help='The Pretrained Model' , default='human_512_2++.hdf5',  required=True)
args = parser.parse_args()


model = load_model(args.model, custom_objects={"awesomeq_loss" : awesomeq_loss, 'dice_loss':dice_loss})
img2 = potrait_extract(model, args.img_path, False, True)
plt.imshow(img2)
plt.show()