# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./raw_image_nonflipped.png "non-flipped"
[image2]: ./raw_image.png "raw images"
[image3]: ./Cropped_image.png "cropped Image"
[image4]: ./Histogram.png "histograms"
[image5]: ./loss.png "loss"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py and mycode-2.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_best.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* dev.py showing the plots and data analytics

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network consists of 5 convolutional layers and 3 fully connected layers. This architecture was chosen based on [Nvidia paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) (lines 121 To 147). After each layer Maxpool of size 2 was used to reduce overfitting (lines 127, 130, 136, 136, 139).

The model includes ELU layers to introduce nonlinearity and Tanh in he last layer(code line 147). The data is normalized in the model using a Keras lambda layer (code line 123).

#### 2. Attempts to reduce overfitting in the model

The model contains max pool layers in order to reduce overfitting (model.py lines 127, 130, 133, 136, 139). Also, early stopping was implemented in teh call back (model.py line 154). 

Since I originally was using more epoch number, I used to define an early stopping in my callbacks as well to prevent overfitting my model.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I decided my data to 80% for training and 20% for validation; and every time the best model based on the validation score was saves as `model_best.h5` (line 150). Also, the final model was saved as `model.h5`, but it was not used since it could be overfitted to the training data (also, its performance was not good in practice). 

Also, some data augmentation was done on the data. First I added the left and right cameras with correction value of 0.2 to the respective steering angle. I also, flipped the images to increase the number of training set. At the end I added a white noise to the 0 steering angle data. The white noise was less than 0.01 and did not change the performance of the ride, but it decreases the bias of the data from zero.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually with teh learning rate of $10^{-5}$ (model.py line 159). The ReduceLROnPlateau function is also used in call backs to monitor the validation loss so that if it doesn't change within 5 time-steps it attempts to reduce it by a factor of 0.8. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of 3 laps of center lane driving and recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architecture was inspired by Nvidia model (lines 121 To 147), consists of 5 convolutional layers and 4 fully connected layers. Since I did not know about the best epoch number, an early stopping was defined and the best model best on the validation data was saved to prevent the overfitting.

This model used ELU and Tanh (only at the last full connected layer) to introduce the non-linearity to the model.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation sets (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added max pool layers after each convolutional layer. Then I used early stooping to prevent the overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There was only one spot that vehicle got little bit close to the line but it maintained to stay on the road. 

To improve the model, I flipped the images and added right and left cameras, in addition to add max pool layers to the model. Also, all the images were cropped by using eras lambda function to prevent the model to decide based on the background and be more focused on the road.

I found out that there is bias toward zero steering angle in the data, which I added a white noise to break the bias toward zero. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

* Normalization layer
* Cropping layer
* Convolution layer with 24 filters and 5x5
* Max Pool layer
* Convolution layer with 36 filters and 5x5
* Max Pool layer
Convolution layer with 48 filters and 5x5
* Max Pool layer
* Convolution layer with 64 filters and 3x3
* Max Pool layer
Convolution layer with 64 filters and 3x3
* Max Pool layer
* Flatten layer
* Fully connected layer with 100 nodes
* Fully connected layer with 50 nodes
* Fully connected layer with 10 nodes
* Fully connected layer with 1 node

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track two using center lane driving and one using the recovery to help the vehicle recovering when it is close to the lanes.

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here are the images from previous figure that were flipped. 

![alt text][image2]
 
For improving the initial results of the model; it was decided to cropped the images. Thus, the model would be based on the road images and not the backgrounds. 

![][image3]

As the figures indicate, for augmenting the data, images from center, right, and left cameras were used for this project. The reason for that is the fact the amount of data and the distribution of data were not good by using inly the center camera. Adding images from left and right camera correct the problem of number of images but the distribution was still biased toward zero. For solving this problem a white noise was added to the zero steering angle data. Since this white noise was so small it did not have significant effect on behavior of the car. The figure below illustrates the distribution of the steering angle data after and before augmentation.

![][image4] 

After the augmentation I had 20316 data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 9 as evidenced by validation loss as well as the early stopping function I used an adam optimizer so that manually training the learning rate wasn't necessary.

The resulted loss rates are in below figure

![][image5]

***Please note that this model was trained on 640x480 resolution and in fast quality.***