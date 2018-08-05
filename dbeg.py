import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

# %% Loading data
lines = []
with open("/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#imagesC = np.empty([3386,160,320,3])
imagesC = []
i = []

j = 0
i = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/' + filename 
    image = cv2.imread(current_path,1)

    #imagef = cv2.flip(image, 1)
    imagesC.append(image)
    #imagesC.append(imagef)
    #imagesC[j,:,:,:] = image
    i.append(image)
    j = j+1


#%% Loading Libraries
import csv
import cv2
import numpy as np
import pandas as pd

#%% Loading data
lines = []
with open("/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/driving_log.csv") as csvfile:
    reader = csv.reader( csvfile )
    for line in reader:
        lines.append( line )

#%% REading the images and steering values
imagesC = []
imagesL = []
imagesR = []
measurmentsC = []
measurmentsL = []
measurmentsR = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/'\
    + filename 
    image = cv2.imread( current_path )
    imagef = cv2.flip(image, 1)
    imagesC.append(image)
    imagesC.append(imagef)
    
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/'\
    + filename 
    image = cv2.imread( current_path )
    imagef = cv2.flip(image, 1)
    imagesL.append(image)
    imagesL.append(imagef)

    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/'\
    + filename 
    image = cv2.imread( current_path )
    imagef = cv2.flip(image, 1)
    imagesR.append(image)
    imagesR.append(imagef)
    
    measurment = float(line[3])
    measurmentsC.append( measurment )
    measurmentsC.append( -measurment)

    measurmentl = measurment+.2
    measurmentsL.append( measurmentl)
    measurmentsL.append(-measurmentl)

    measurmentr = measurment-.2
    measurmentsR.append( measurmentr)
    measurmentsR.append(-measurmentr)
    
    
X_train = []
X_train.extend(imagesC)
X_train.extend(imagesL)
X_train.extend(imagesR)
X_train = np.array(X_train)

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

## Adding noise to the measurments for reducing the abosoute zero values and
## increasing the variation of the data 
plt.figure("Histograms", figsize = (13,8))
plt.subplot(1,3,1)
plt.hist(measurmentsC)
plt.title("Only Centeral Camera")
plt.ylim(0,12000)

plt.subplot(1,3,2)
measurments = []
measurments.extend(measurmentsC)
measurments.extend(measurmentsL)
measurments.extend(measurmentsR)
plt.hist(measurments)
plt.title("Using all Cameras")
plt.ylim(0,12000)

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

plt.subplot(1,3,3)
plt.hist(measurments)
plt.title("Added White Noise")
plt.ylim(0,12000)