# Semantic Segmentation with Fully Convolutional Networks

### Introduction
Label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
You can use semantic_segmantation_env.yml which is prepared for this project or 

Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.

## Fully Convolutional Networks
<img width="800" alt="Scheme of Semantic Segmantation with Fully Convolutional Network - https://medium.com/@subodh.malgonde/transfer-learning-using-tensorflow-52a4f6bcde3e" src="/imgs/FCN 1.png">

If we want to know where is an object in a given image fully connected layers won't give us what we need because fully connected layers cannot store spatial information. Replacing fully connected layers with 1x1 Convolutional layers will make it possible. Since the images size is not important for convolutional operations, "any size of image" can be fed to network. 

3 techniques used by FCN's:

1. Replace fully connected layers with 1x1 convolutional layers
<img width="400" alt="1x1 Convolutional Layers" src="/imgs/1x1 convolutional.JPG">

2. Upsampling through the use of transposed convolutional layers
<img width="400" alt="Upsampling" src="/imgs/upsampling.JPG">

3. Skip Connections
<img width="400" alt="Skip Connections" src="/imgs/skip_connections.JPG">





<img width="800" alt="Semantic Segmantation with Fully Convolutional Network" src="/imgs/semantic_segmantation.gif">

