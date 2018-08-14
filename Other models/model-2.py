#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:28:41 2018

@author: Arash
"""

#%% Loading Libraries
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn
import matplotlib.pyplot as plt
plt.switch_backend('agg')


#%% Loading data

# Reading data addresses
lines = []  # Addresses
with\
open( "/home/workspace/CarND-Behavioral-Cloning-P3/data/driving_log.csv" ) as\
 csvfile:
    reader = csv.reader( csvfile )
    for line in reader:
        lines.append( line )

# Loading images and steerings measurment        
imagesC = []        # Images from centeral camera
imagesL = []        # Images from left camera 
imagesR = []        # Images from right camera
measurmentsC = []   # Steering measurents respect to centeral camera
measurmentsL = []   # Steering measurents respect to left camera
measurmentsR = []   # Steering measurents respect to right camera
for line in lines:
    
    # Loading images from centeral camera
    source_path = line[ 0 ]
    filename = source_path.split( '/' )[ -1 ]
    current_path =\
    '/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/' +\
    filename
    image = cv2.imread( current_path )  # Loading the image
    imagef = cv2.flip( image, 1 )       # Flipping the image
    imagesC.append( image )             # Saving the images
    imagesC.append( imagef )            # Saving the flipped images
    
    # Loading images from left camera
    source_path = line[ 1 ]
    filename = source_path.split( '/' )[ -1 ]
    current_path =\
    '/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/' +\
    filename
    image = cv2.imread( current_path )  # Loading the image
    imagef = cv2.flip( image, 1 )       # Flipping the image
    imagesL.append( image )             # Saving the images
    imagesL.append( imagef )            # Saving the flipped images

    # Loading images from right camera
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = '/home/workspace/CarND-Behavioral-Cloning-P3/data/IMG/' +\
    filename
    image = cv2.imread( current_path )  # Loading the image
    imagef = cv2.flip( image, 1 )       # Flipping the image
    imagesR.append( image )             # Saving the images
    imagesR.append( imagef )            # Saving the flipped images
    
    # Loading steering measurments for centeral images
    measurment = float(line[3])
    measurmentsC.append( measurment )
    measurmentsC.append( -measurment )
    
    # Relative steering measurements for left camera
    measurmentl = measurment+.2
    measurmentsL.append( measurmentl )
    measurmentsL.append(-measurmentl )

    # Relative steering measurements for right camera
    measurmentr = measurment-.2
    measurmentsR.append( measurmentr )
    measurmentsR.append( -measurmentr )
    
# Contatnating images as the training set    
X_train = []
X_train.extend( imagesC )
X_train.extend( imagesL )
X_train.extend( imagesR )
X_train = np.array( X_train )

## Adding white noise to the zero steering measurements to reduce the bia of
## data
measurmentsC = np.array( measurmentsC )
measurmentsC[ measurmentsC == 0 ] =\
measurmentsC[ measurmentsC == 0 ] +\
( .0001 * np.random.randn( measurmentsC[ measurmentsC == 0 ].shape[ 0 ] ) )

measurmentsR = np.array( measurmentsR )
measurmentsR[ measurmentsR == -.2 ] =\
measurmentsR[ measurmentsR == -.2 ]+\
(.0001 * np.random.randn( measurmentsR[ measurmentsR == -.2 ].shape[ 0 ] ) )

measurmentsL = np.array( measurmentsL )
measurmentsL[ measurmentsL == .2 ] =\
measurmentsL[ measurmentsL == .2 ]+\
( .0001 * np.random.randn( measurmentsL[ measurmentsL == .2 ].shape[ 0 ] ) )

# Concatening the steering measurments for the training set
measurments = []
measurments.extend( measurmentsC )
measurments.extend( measurmentsL )
measurments.extend( measurmentsR )
y_train = np.array( measurments )

'''
smpl_im = np.random.randint( 0, len( imagesC ) )
plt.figure("images", figsize = (13,8))
plt.subplot(3,3,1)
plt.title("Raw Left Camera")
plt.imshow(X_train[ smpl_im+len(imagesC), :, :, : : -1])

plt.subplot(3,3,2)
plt.title("Raw Center Camera")
plt.imshow(X_train[ smpl_im, :, :, : : -1])

plt.subplot(3,3,3)
plt.title("Raw Right Camera")
plt.imshow(X_train[ smpl_im+2*len(imagesC), :, :, : : -1])

plt.subplot(3,3,4)
plt.title("Cropped Left Camera")
plt.imshow(X_train[ smpl_im+len(imagesC), 75:135, :, : : -1])

plt.subplot(3,3,5)
plt.title("Cropped Center Camera")
plt.imshow(X_train[ smpl_im, 75:135, :, : : -1])

plt.subplot(3,3,6)
plt.title("Cropped Right Camera")
plt.imshow(X_train[ smpl_im+2*len(imagesC), 75:135, :, : : -1])

plt.subplot(3,3,7)
plt.title("Scaled Left Camera")
plt.imshow(X_train[ smpl_im+len(imagesC), 75:135, :, : : -1]/255.0-.5)

plt.subplot(3,3,8)
plt.title("Scaled Center Camera")
plt.imshow(X_train[ smpl_im, 75:135, :, : : -1]/255.0-.5)

plt.subplot(3,3,9)
plt.title("Scaled Right Camera")
plt.imshow(X_train[ smpl_im+2*len(imagesC), 75:135, :, : : -1]/255.0-.5)
'''

## Adding noise to the measurments for reducing the abosoute zero values and
## increasing the variation of the data 
'''
plt.figure("Histograms", figsize = (13,8))
plt.subplot(1,3,1)
plt.hist(measurmentsC)
plt.title("Only Centeral Camera")
plt.ylim(0,12000)
'''

'''
plt.subplot(1,3,2)
measurments = []
measurments.extend(measurmentsC)
measurments.extend(measurmentsL)
measurments.extend(measurmentsR)
plt.hist(measurments)
plt.title("Using all Cameras")
plt.ylim(0,12000)
'''
'''
measurmentsC = np.array(measurmentsC)
measurmentsC[measurmentsC==0] = measurmentsC[measurmentsC==0]+\
(.00001*np.random.randn(measurmentsC[measurmentsC==0].shape[0]))

measurmentsR = np.array(measurmentsR)
measurmentsR[measurmentsR==-.2] = measurmentsR[measurmentsR==-.2]+\
(.00001*np.random.randn(measurmentsR[measurmentsR==-.2].shape[0]))

measurmentsL = np.array(measurmentsL)
measurmentsL[measurmentsL==.2] = measurmentsL[measurmentsL==.2]+\
(.00001*np.random.randn(measurmentsL[measurmentsL==.2].shape[0]))



measurments = []
measurments.extend(measurmentsC)
measurments.extend(measurmentsL)
measurments.extend(measurmentsR)

y_train = np.array(measurments)
'''
'''
plt.subplot(1,3,3)
plt.hist(measurments)
plt.title("Added White Noise")
plt.ylim(0,12000)
'''
from sklearn.utils import shuffle
'''
def generator(X_train, y_train, batch_size = 32):
    X = []
    y = []
    X_train, y_train = shuffle(X_train, y_train)
    while True:
        for i in range(len(y_train)):
            xx = X_train[i,:,:,:]
            yy = y_train[i]
            X.append( xx )
            y.append( yy )
            if len( y ) == batch_size:
                yield (np.array(X), np.array(y))
                '''
def generator(X_train, y_train, batch_size = 32):
    X_train, y_train = shuffle(X_train, y_train)
    X = np.zeros([32,160,320,3])
    y = np.zeros([32,])
    k = 0
    while True:
        for i in range(len(y_train)):
            X[k,:,:,:] = X_train[i,:,:,:]
            y[k] = y_train[i]
            k = k+1
            if k == batch_size:
                k = 0
                yield shuffle(X, y)
        
##
#
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Activation, Cropping2D, SpatialDropout2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Lambda(lambda x:x/255-.5, input_shape = (160,320,3),output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(6, 5, 5, subsample=(1, 1), border_mode="valid", activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
model.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="valid", activation="elu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))


model.add(Flatten())
model.add(Dense(120, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(65, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(20, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(1))

checkpoint = ModelCheckpoint("model_best.h5", monitor='val_loss', verbose=1, save_best_only=True,mode='auto')
    
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6)    
    
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    
outputs=[checkpoint, reduce_lr, early_stop]
#model.compile(loss='mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs=30)
print('Model is runing...')
print('==================')
#model, outputs = driveNet()
model.compile(loss='mse',optimizer='adam')
print('train!')
X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train, test_size=0.2)
#le = len(y_val)
#X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
#X_val, y_val = sklearn.utils.shuffle(X_val, y_val)
train = generator(X_tra, y_tra, batch_size = 32)
val = generator(X_val, y_val, batch_size = 32)
history = model.fit_generator( train, validation_data = val, nb_epoch= 10, verbose = 1, nb_val_samples=len(X_val), samples_per_epoch = len(X_tra), callbacks = outputs )
### plot the training and validation loss for each epoch
  # summarize history for loss
print ("Plotting losses ... ")
#plot_model(model, to_file='model.png')
   
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()

model.save('model.h5')

