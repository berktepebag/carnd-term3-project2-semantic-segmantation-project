
### *** Archieved: As this repo is and will not updated anymore, it is archieved. ***

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

1. Replace fully connected layers with 1x1 convolutional layers:

This will let us have 4D tensor instead of a flattened 2D tensor, so spatial information will be preserved.
<img width="400" alt="1x1 Convolutional Layers" src="/imgs/1x1 convolutional.JPG">


2. Upsampling through the use of transposed convolutional layers
<img width="400" alt="Upsampling" src="/imgs/upsampling.JPG">

Since we want last layer have the same dimension with the input we have to upsample layers. Stride will give the multiplier of previous layer to next layer. So if we want to upsample 3x3 input to 6x6, stride should be 6/3 = 2. 

3. Skip Connections
<img width="400" alt="Skip Connections" src="/imgs/skip_connections.JPG">

Each layer of convolution network carry different informations such as edges, shapes... If we upsample from only last layer of encoder part, we will lose information from these layers. By skip connections we get information from previous convolutional layers into upsampling.

### Fully Convoltional Layout
<img width="800" alt="Fully Convoltional Layout" src="/imgs/FCN_layout.JPG">

Fully Convoltional Networks consists two parts, Encoder and Decoder. Since encoder part is similar to any convolutional network we can use pre-trained ones with transfer learning and add decoder part to create a FCN.

Getting the input third, fourth and last layer from encoder part for skipping.
```python
    w1 = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    w3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    w4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    w7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
```

Since we already have a pre-trained network we do not have to run all over the network. We will prevent encoder layers from being trained.
```python
    vgg_layer7_out = tf.stop_gradient(vgg_layer7_out)
    vgg_layer4_out = tf.stop_gradient(vgg_layer4_out)
    vgg_layer3_out = tf.stop_gradient(vgg_layer3_out)
```

Applying 1x1 convolution and transpose. We will repeat this for each layer we are using from encoder.
```python
    # L7 -> L4    
    # 1x1 Convolution of vgg layer 7
    l7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes , 1, padding='SAME'
        ,kernel_initializer= tf.random_normal_initializer(stddev=0.01)
        , kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-03))     
    l7_conv_1x1_transpose = tf.layers.conv2d_transpose(l7_conv_1x1, num_classes, 4, strides = 2, padding = 'SAME'
        , kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-03))

    # 1x1 Convolution of vgg layer 4
    l4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes , 1, padding='SAME'
        ,kernel_initializer= tf.random_normal_initializer(stddev=0.01)
        , kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-03))   

    #Combine the output of two layers. The first output is the output of the current layer.
    #The second output is the output of a layer further back in the network, typically a pooling layer.
    l7_l4_sum = tf.add(l7_conv_1x1_transpose,l4_conv_1x1)
```
## Result: 

| Epoch         | Batch           | Learning Rate  |
| ------------- |---------------| -----|
|2 |4 | 0.01|
|2 |64| 0.01|
|2 |16 | 0.01|
|10 |16| 0.0001|
|10 |16| 0.001|
|15 |16| 0.001|
|5 |8| 0.001|
|20 |8| 0.001|
|50 |8| 0.001|
|20 |2| 0.001|
|10 |6| 0.0008|
|40 |6| 0.0008|

In the vanilla neural networks we prefer 64-128 batches to achieve low loss. Unlike vanilla NN Fully convolutional NN's are not good with high batches. So lowered batches over my trials.

<img width="800" alt="High batch causes artifacts" src="/imgs/high_batch_problems.png">

When learning rate is higher, smoothness of the classified road decreases. 

<img width="800" alt="High batch causes artifacts" src="/imgs/learning_rate_diff.png">

Larger epochs also helps with better learning but after few epochs accuracy does not increase.

### Result Images:
https://drive.google.com/open?id=1AiOxwipxF37PQjiGBcNkop_44ail7w7O

<img width="800" alt="Semantic Segmantation with Fully Convolutional Network" src="/imgs/semantic_segmantation.gif">

