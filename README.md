# Human Portrait Segmentation
* The pretrained model : [The Pretrained Model ](https://drive.google.com/open?id=1SWJ4CFEjez8EoZeGqeSkGgjiavIqXU3y)
* To test pretrained model :<b> python test.py -i 'path_of_image.png' -m 'pretrained_model_path.hdf5'</b>
* To run real-time on webcam :<b> python webcam.py -m 'pretrained_model_path.hdf5'</b>
## Some Test Results
<p align="center"> 
<img src="https://github.com/saitakturk/portrait_segmentation/blob/master/test_images/test1.png">
</p>

![Image](https://github.com/saitakturk/portrait_segmentation/blob/master/test_images/test2.png)
![Image](https://github.com/saitakturk/portrait_segmentation/blob/master/test_images/test3.png)
## Dataset and Preprocessing
* The dataset link : [The Dataset ](https://drive.google.com/open?id=1JMuEA1qOUTYeJd8AWCvmMjFZfNbA1-DP)
* The dataset originally contains 1597 images and masks. The dataset original :  [The Dataset Original](http://xiaoyongshen.me/webpage_portrait/index.html)
* The given dataset was created by random cropping of an image 5 times. The new image size is 512x512x3.
* The training set contains 6985 images and corresponding masks, the validation set has 500 images and masks.

## Data Augmentation
####  Brightness Augmentation
* To make the model work well on different brightness level, I used brightness augmentation. The color space of images is changed from RGB to HSV color space. “value”(brightness) layer of the image is randomized to create images with different brightness level. Then, I changed color HSV color space to RGB color space that could be used for deep learning model.

####  Image Quality Augmentation
* Since the project aim was to create real-time semantic segmentation model, I should make the model generalizable to different camera and image quality. I used OpenCV library IMWRITE_JPEG_QUALITY function to reduce quality of the image with randomized value. 

## Model
* I used U-Net architecture because it was easy to implement, powerful to get high accuracy and fast enough to work real-time on GPUs.
U-Net architecture is vastly used in the area of biomedical image segmentation. U-Net architecture consists of two main parts. First part of the architecture encodes the image to get high level features. Second part of the architecture decodes features that got from first part of the architecture. In U-Net architecture, there are residual connections between encoder layer and decoder layers to use high level features in decoder part. Since I want to create model that could work real-time. For encoder part, I used MobileNetV2 that consists of Convolution, Depthwise Convolution and Max Pooling layers. For decoder part, I used Transposed Convolution and Upsampling layers that are inverse of the Max Pooling and Convolution operation.

## Loss 
* For semantic segmentation tasks, the one of the most important metric is the intersection over union( IoU ) that is the ratio of intersection area of target mask and output mask over union area of target mask and output mask. Therefore, in the literature, dice loss is  vastly used to calculate IoU loss of the output mask and target mask. However, I want to calculate loss of the model, not only for IoU ratio but also similarity between pixel values of the output and mask with using binary cross entropy loss that the dataset have two classes. Therefore, I used combination of two losses that “Dice loss + Binary Cross Entropy Loss”. I trained our network with only dice loss and combination of the two losses. I observed that the combination of the losses got higher accuracy and meaningful results in terms of the localizing the object.

## Some Experiments and Observations
* I experimented with number of output layers, I created 5 output images with resizing images into intermediate layers. I observed that the multiple loss and backpropagation  could not converged since there are multiple local objectives. However, I assume that it will not be problem for big model. The more residual connections helps to increase the IoU accuracy on test set.
