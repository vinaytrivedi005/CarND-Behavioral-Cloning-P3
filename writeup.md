
# **Behavioral Cloning**

This project is about clonning the behaviour of human driver to drive around a track on simulator.


```python
# import libraries here

import pickle
import os
from urllib.request import urlretrieve
from zipfile import ZipFile
import random

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

print('All libraries imported successfully')
```

    Using TensorFlow backend.
    

    All libraries imported successfully
    

I am using provided sample data as hardware limitation of my desktop did not allow me to run the simulator properly.


```python
# downloading data

def download(url, file):
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')
        #Unzip the downloaded file to get pickled data
        zip = ZipFile('data.zip')
        zip.extractall()

# Downloading the training and test dataset.
download('https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip', 'data.zip')

# Wait until you see that all files have been downloaded.
print('All files downloaded.')
```

    All files downloaded.
    


```python
# reading measurement file

lines = []

with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)
        
print('mesurement file read successful')
```

    mesurement file read successful
    

### Reading and augmenting data

Here I read sample data and used 30% of right images and 30% left images and 80% center images. I used right and left images to reduce biased towards center. 


```python
# reading images and measurements from measurement file

images = []
measurements = []

for line in lines[1:]:
    r = random.uniform(0,1)
    if r <= 0.3:
        source_path = line[1]
        file_name = source_path.split('/')[-1]
        current_path = './data/IMG/' + file_name
        image = cv2.imread(current_path)
        measurement = float(line[3]) + 0.25
        images.append(image)
        measurements.append(measurement)
    if r > 0.2 and r < 0.8:
        source_path = line[0]
        file_name = source_path.split('/')[-1]
        current_path = './data/IMG/' + file_name
        image = cv2.imread(current_path)
        measurement = float(line[3])
        images.append(image)
        measurements.append(measurement)
    if r >= 0.7:
        source_path = line[2]
        file_name = source_path.split('/')[-1]
        current_path = './data/IMG/' + file_name
        image = cv2.imread(current_path)
        measurement = float(line[3]) - 0.25
        images.append(image)
        measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print('data reading completed.')
```

    data reading completed.
    

### Model architecture

I am using same architecture used by NVIDIA's team. I increased number of filter in last convolution layer from 64 to 80. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.
The model used an adam optimizer, so the learning rate was not tuned manually.

In the first few runs, my model was going outside the marked lines, to overcome this I added more filters in last layer so that more features can be extracted.

After that, it was driving fine but not taking turns properly. THis implies that training data for curves is not enough. To overcome this, I added 30% left and 30% right camara images to make model learn how to recover from left and right and drive back to center. It was also biased towards center so I reduced 20% center image data to overcome that biased behavior.

I also split data to 80% training and 20% validation to check the accuravy of model.

I used mean squared error as loss function and 5 epoch to train my model.

The test run video on simulater is included in submission with name run1.mp4 .

My architecture is as below:

[//]:<> ( Layer (type)	Output Shape	Param #	Connected to                                     )
[//]:<> ( lambda_2 (Lambda)	(None, 160, 320, 3)	0	lambda_input_2[0][0]                         )    
[//]:<> ( cropping2d_2 (Cropping2D)	(None, 65, 320, 3)	0	lambda_2[0][0]                       )      
[//]:<> ( convolution2d_7 (Convolution2D)	(None, 31, 158, 24)	1824	cropping2d_2[0][0]       )                 
[//]:<> ( convolution2d_8 (Convolution2D)	(None, 27, 154, 24)	14424	convolution2d_7[0][0]    )                                                 
[//]:<> ( convolution2d_9 (Convolution2D)	(None, 12, 75, 36)	21636	convolution2d_8[0][0]    )                                              
[//]:<> ( convolution2d_10 (Convolution2D)	(None, 5, 37, 48)	15600	convolution2d_9[0][0]    )                                
[//]:<> ( convolution2d_11 (Convolution2D)	(None, 3, 35, 64)	27712	convolution2d_10[0][0]   )                                
[//]:<> ( convolution2d_12 (Convolution2D)	(None, 1, 33, 80)	46160	convolution2d_11[0][0]   )                                                                      
[//]:<> ( flatten_2 (Flatten)	(None, 2640)	0	convolution2d_12[0][0]                       )            
[//]:<> ( dense_5 (Dense)	(None, 100)	264100	flatten_2[0][0]                                  ) 
[//]:<> ( activation_4 (Activation)	(None, 100)	0	dense_5[0][0]                                )   
[//]:<> ( dense_6 (Dense)	(None, 50)	5050	activation_4[0][0]                               )    
[//]:<> ( activation_5 (Activation)	(None, 50)	0	dense_6[0][0]                                )   
[//]:<> ( dense_7 (Dense)	(None, 10)	510	activation_5[0][0]                                   )
[//]:<> ( activation_6 (Activation)	(None, 10)	0	dense_7[0][0]                                )   
[//]:<> ( dense_8 (Dense)	(None, 1)	11	activation_6[0][0]                                   )
![image.png](attachment:image.png)

##### Total params: 397,027
##### Trainable params: 397,027
##### Non-trainable params: 0


```python
# generating model

model = Sequential()

# Pre-processing data
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# Layer 1: convolution layer, input 65 x 320 x 3, output 63 x 318 x 32
#model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))

model.add(Convolution2D(24, 5, 5, activation="relu"))

# Layer 2: convolution layer
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))

# Layer 3: convolution layer
model.add(Convolution2D(48, 3, 3, subsample=(2,2), activation="relu"))

model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(80, 3, 3, activation="relu"))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation("relu"))
#model.add(Dropout(0.3))

model.add(Dense(50))
model.add(Activation("relu"))
#model.add(Dropout(0.35))

model.add(Dense(10))
model.add(Activation("relu"))
#model.add(Dropout(0.2))

model.add(Dense(1))

model.summary()


```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    lambda_2 (Lambda)                (None, 160, 320, 3)   0           lambda_input_2[0][0]             
    ____________________________________________________________________________________________________
    cropping2d_2 (Cropping2D)        (None, 65, 320, 3)    0           lambda_2[0][0]                   
    ____________________________________________________________________________________________________
    convolution2d_7 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_2[0][0]               
    ____________________________________________________________________________________________________
    convolution2d_8 (Convolution2D)  (None, 27, 154, 24)   14424       convolution2d_7[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_9 (Convolution2D)  (None, 12, 75, 36)    21636       convolution2d_8[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_10 (Convolution2D) (None, 5, 37, 48)     15600       convolution2d_9[0][0]            
    ____________________________________________________________________________________________________
    convolution2d_11 (Convolution2D) (None, 3, 35, 64)     27712       convolution2d_10[0][0]           
    ____________________________________________________________________________________________________
    convolution2d_12 (Convolution2D) (None, 1, 33, 80)     46160       convolution2d_11[0][0]           
    ____________________________________________________________________________________________________
    flatten_2 (Flatten)              (None, 2640)          0           convolution2d_12[0][0]           
    ____________________________________________________________________________________________________
    dense_5 (Dense)                  (None, 100)           264100      flatten_2[0][0]                  
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 100)           0           dense_5[0][0]                    
    ____________________________________________________________________________________________________
    dense_6 (Dense)                  (None, 50)            5050        activation_4[0][0]               
    ____________________________________________________________________________________________________
    activation_5 (Activation)        (None, 50)            0           dense_6[0][0]                    
    ____________________________________________________________________________________________________
    dense_7 (Dense)                  (None, 10)            510         activation_5[0][0]               
    ____________________________________________________________________________________________________
    activation_6 (Activation)        (None, 10)            0           dense_7[0][0]                    
    ____________________________________________________________________________________________________
    dense_8 (Dense)                  (None, 1)             11          activation_6[0][0]               
    ====================================================================================================
    Total params: 397,027
    Trainable params: 397,027
    Non-trainable params: 0
    ____________________________________________________________________________________________________
    


```python
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model1.h5')
```

    Train on 7724 samples, validate on 1931 samples
    Epoch 1/5
    7724/7724 [==============================] - 25s - loss: 0.0260 - val_loss: 0.0277
    Epoch 2/5
    7724/7724 [==============================] - 25s - loss: 0.0213 - val_loss: 0.0235
    Epoch 3/5
    7724/7724 [==============================] - 25s - loss: 0.0203 - val_loss: 0.0248
    Epoch 4/5
    7724/7724 [==============================] - 25s - loss: 0.0194 - val_loss: 0.0262
    Epoch 5/5
    7724/7724 [==============================] - 25s - loss: 0.0182 - val_loss: 0.0242
    
