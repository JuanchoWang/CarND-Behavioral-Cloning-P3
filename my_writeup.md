# **Behavioral Cloning** 

## My Writeup for Submission

#### Hereby I describe how I finished the 3rd project of this Nanodegree program and how I achieved all the given goals..

---

**Created on 18.07.2020**
First writeup finished.

[//]: # (Image References)

[image1]: ./figure/network_architecture.png "Model Architecture"

---

### Network Architecture and Implementation Details

#### 1. Convolutional Neural Network modified from NVIDIA's End-to-End Learning Network

My network architecture is directly modified from the network proposed in NVIDIA's [End-to-End Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/). The main parts, the convolutional layers, are almost kept intact but the last several fully connected layers are slightly changed.

Specifically, my version consists of 5 convolutional layers for feeding the inputs. They are followed by two FC-layers, one has 100 neurons and the other one has 25 neurons. Inbetween, a dropout with a probabilty of 50% is added to learn more redundant features. The final layer is a regression with mean square erorr as loss function. (model.py lines 79-88) 

| Layer		|  Filter/Neuron	| Stride	| Input size	|
|:---------:|:-----------------:|:---------:|:-------------:|
| Conv1		|  24x5x5			|  2x2		|  3x90x320		|
| Conv2    	|  36x5x5 			|  2x2		| 24x43x158		|
| Conv3    	|  48x5x5 			|  2x2		| 36x20x77		|
| Conv4    	|  64x3x3 			|  -		| 48x8x37		|
| Conv5    	|  64x3x3 			|  -		| 64x3x18		|
| FC-1    	|  100  			|  -		| 64x1x8		|
| FC-2    	|  25  				|  -		| 512			|

![alt text][image1]

#### 2. Dataset Preparation

All training data I collected is multi-camera images by center driving. For that, I recorded 3 laps in counter-clockwise on track one and 1 lap in clockwise. For data augmentation, I also flipped the center image horizontally. After the following normalization, the images are fed into the CNN as input.

Due that I recorded all the training data in the workspace, I haven't saved the images locally to show them here.

Before passing the dataset to my network, they are shuffled first and 20% of them will be treated as a validation set. 

#### 3.Training Strategy

I created a generator for both training and validation data to feed my network with input. In this way, I managed to save the memory and only retrieve the data when I need it. (Not sure if I have really done it right. It seemed I had still loaded all the images. See model.py lines 52-66)

All the layers of my network except the last regression layer adopt RELU as activations. To combat overfitting, a dropout is also contained.

The model used an adam optimizer, so the learning rate was not tuned manually. The batch size is 32 and I chose a training iteration with 10 epochs, because I had seen a validation error increase after 10 epochs. An early stop is a must.

In practice, after 10 epochs of training I was able to get a model with a validation error less than 0.02. The training error was even smaller and would become further smaller if I trained it more.


### Drawback and Improvement

#### Data Collection and Performance on Track 2

I didn't even try to record a recovery driving on track one. Since I had chosen an appropriate network architecture, the model has already worked for the first time even only training with center camera images.

For track two, I also recorded a 3 lap center driving. As described in the Rubrics, it is much more difficult and I myself even took lots of time to finish one lap recording. The whole autonomous driving for testing looked mostly not bad but the vehicle would drive off the road in a sharp curve. I guess I didn't perform well there during training and caused such consequences. I also reduced the target speed from 9 to 5 in drive.py but it still couldn't make me succeed in those turns.
