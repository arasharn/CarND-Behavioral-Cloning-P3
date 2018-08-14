import csv
import cv2
import numpy as np
#%% Loading data
print('Loading data...')
# Reading data addresses
lines = []  # Addresses
with open( "/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/driving_log.csv" ) as csvfile:
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
    '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/' +\
    filename
    image = cv2.imread( current_path )  # Loading the image
    imagef = cv2.flip( image, 1 )       # Flipping the image
    imagesC.append( image )             # Saving the images
    imagesC.append( imagef )            # Saving the flipped images
    
    # Loading images from left camera
    source_path = line[ 1 ]
    filename = source_path.split( '/' )[ -1 ]
    current_path =\
    '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/' +\
    filename
    image = cv2.imread( current_path )  # Loading the image
    imagef = cv2.flip( image, 1 )       # Flipping the image
    imagesL.append( image )             # Saving the images
    imagesL.append( imagef )            # Saving the flipped images

    # Loading images from right camera
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = '/Users/Arash/GitHub/CarND-Behavioral-Cloning-P3/data/IMG/' +\
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
#%%  
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
y_train = []
y_train.extend( measurmentsC )
y_train.extend( measurmentsL )
y_train.extend( measurmentsR )
y_train = np.array( y_train )

X = np.zeros([32,160,320,3])
y = np.zeros([32,])
k = 0
c = 0
for i in range(len(y_train)):
    X[k,:,:,:] = X_train[i,:,:,:]
    y[k] = y_train[i]
    k = k+1
    if k == 32:
        k = 0
        c = c+1
    #print(np.array(X_train[i]).shape)
    #x = X_train[i]
    #y = y_train[ i ]
    #X.append( x )
    #y.append( y )
    

'''
from sklearn.utils import shuffle
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
print("running generator...")  
xx,yy = generator(X_train,y_train, batch_size = 32)
print('done')
'''