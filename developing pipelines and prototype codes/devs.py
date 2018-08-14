#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:45:59 2018

@author: Arash
"""

#%%
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout



#%%
lines = []
with open( "/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/driving_log.csv" ) as csvfile:
    reader = csv.reader( csvfile )
    for line in reader:
        lines.append( line )

imagesC = []
imagesL = []
imagesR = []
measurmentsC = []
measurmentsL = []
measurmentsR = []
for line in lines:
    image = cv2.imread( line[ 0 ] )
    imagef = cv2.flip(image,1)
    imagesC.append( image )
    imagesC.append(imagef)

    image = cv2.imread(line[1])
    imagef = cv2.flip(image, 1)
    imagesL.append(image)
    imagesL.append(imagef)

    image = cv2.imread(line[2])
    imagef = cv2.flip(image,1)
    imagesR.append(image)
    imagesR.append(imagef)

    measurment = float( line[ 3 ] )
    '''if measurment == 0:
        measurment = measurment+np.random.rand(1)'''
    measurmentsC.append( measurment )
    measurmentsC.append( -measurment)

    measurmentl = measurment+.2
    measurmentsL.append( measurmentl)
    measurmentsL.append(-measurmentl)

    measurmentr = measurment-.2
    measurmentsR.append( measurmentr)
    measurmentsR.append(-measurmentr)

'''
X_train = np.array( images )
y_train = np.array( measurments )
'''
imagesC = np.array(imagesC)
imagesR = np.array(imagesR)
imagesL = np.array(imagesL)

images = []
images.extend(imagesC)
images.extend(imagesL)
images.extend(imagesR)

measurmentsC_before =measurmentsC
measurmentsC = np.array(measurmentsC)
measurmentsC[measurmentsC==0] = measurmentsC[measurmentsC==0]+(.0001*np.random.randn(measurmentsC[measurmentsC==0].shape[0]))



measurmentsR_before =measurmentsR
measurmentsR = np.array(measurmentsR)
measurmentsR[measurmentsR==-.2] = measurmentsR[measurmentsR==-.2]+(.0001*np.random.randn(measurmentsR[measurmentsR==-.2].shape[0]))

measurmentsL_before =measurmentsL
measurmentsL = np.array(measurmentsL)
measurmentsL[measurmentsL==.2] = measurmentsL[measurmentsL==.2]+(.0001*np.random.randn(measurmentsL[measurmentsL==.2].shape[0]))


measurments = []
measurments.extend(measurmentsC)
measurments.extend(measurmentsL)
measurments.extend(measurmentsR)

measurments_before = []
measurments_before.extend(measurmentsC_before)
measurments_before.extend(measurmentsL_before)
measurments_before.extend(measurmentsR_before)

plt.figure( "Histogram", figsize = (13,8) )
plt.subplot(1,3,1)
plt.title("Center Camera")
plt.hist(measurmentsC)#, np.arange(-1,1,.25))
plt.ylim(0,12000)
plt.subplot(1,3,2)
plt.title("Added side cameras")
plt.hist( measurments_before)#, np.arange(-1,1,.25))
plt.ylim(0,12000)
plt.subplot(1,3,3)
plt.title("Added white noise")
plt.hist(measurments)#, np.arange(-1,1,.25))
plt.ylim(0,12000)
plt.show()

smpl_im = np.random.randint( 0, len( imagesC ) )
plt.figure( "raw image")
plt.subplot( 2, 3, 2 )
plt.title( "Center" )
plt.imshow( np.array(imagesC[ smpl_im, :, :, : : -1 ]) )
plt.subplot( 2, 3, 1 )
plt.title( "Left" )
plt.imshow( np.array(imagesL[ smpl_im, :, :, : : -1 ]) )
plt.subplot( 2, 3, 3 )
plt.title( "Right" )
plt.imshow( np.array(imagesR[ smpl_im, :, :, : : -1 ]) )

plt.subplot( 2, 3, 5 )
plt.title( "Flipped Center" )
flp = np.array(cv2.flip(imagesC[ smpl_im, :, :, :],1))
plt.imshow( np.array(flp[ :, :, : : -1 ]) )
plt.subplot( 2, 3, 4 )
plt.title( "Flipped Left" )
flp = np.array(cv2.flip(imagesL[ smpl_im, :, :, :],1))
plt.imshow( np.array(flp[ :, :, : : -1 ]) )
plt.subplot( 2, 3, 6 )
plt.title( "Flipped Right" )
flp = np.array(cv2.flip(imagesR[ smpl_im, :, :, :],1))
plt.imshow( np.array(flp[ :, :, : : -1 ]) )
plt.show()

#X_train = X_train[ :, 45 : 135,:,: ]

plt.figure( "Cropped image")
plt.subplot( 1, 3, 2 )
plt.title( "Center" )
plt.imshow( imagesC[ smpl_im, 75 : 135, :, : : -1 ])
plt.subplot( 1, 3, 1 )
plt.title( "Left" )
plt.imshow( imagesL[ smpl_im, 75 : 135, :, : : -1 ])
plt.subplot( 1, 3, 3 )
plt.title( "Right" )
plt.imshow( imagesR[ smpl_im, 75 : 135, :, : : -1 ])
plt.show()


y_train = np.array(measurments)
X_train = np.array(images)
#%% Model
'''model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(160,320,3),output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((45,15),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
model.add(Dropout(0.6))
model.add(Dense(50))
model.add(Dropout(0.6))
model.add(Dense(10))
model.add(Dense(1))




model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train,y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2)

from keras.models import Model

print ("Training the model ...")
history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, \
                    callbacks=outputs, validation_data=(X_validation, y_validation), shuffle=False)

print("Plotting losses ... ")
plot(model, to_file='model.png')

# list all data in history
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



model.save('model.h5')
print('done')
los'''
'''t = [0.0344,0.0254,.0225,0.0210,0.0200,0.0194,0.0188,0.0183,0.0179,0.0175]
v = [0.0275,0.0238,0.0220,0.0210,0.0204,0.0197,0.0194,0.0191,.0188,0.0186]
plt.figure()
plt.plot(t)
plt.plot(v)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()'''
