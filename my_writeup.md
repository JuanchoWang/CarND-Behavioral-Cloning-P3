# **Behavioral Cloning** 

## My Writeup for Submission

#### Hereby I describe how I finished the 3rd project of this Nanodegree program and how I achieved all the given goals..

---

**Created on 18.07.2020**
First writeup finished.


[//]: # (Image References)

[image1]: ./figure/network_architecture.png "Model Architecture"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


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

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
