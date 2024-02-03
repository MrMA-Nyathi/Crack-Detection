# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 03:42:54 2024

@author: mnyathi
"""

#----------IMPORTING LIBRARIES---------------
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#changing working directory
os.chdir("C:/Users/mnyathi/Documents/nya-cnn/")
#importing base libraries
import tensorflow as tf 
from tensorflow import keras 

#importing libraries for image processing and loading data

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.applications.vgg16 import preprocess_input

#importing libraries for building and training the model

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

#from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from imblearn.over_sampling import RandomOverSampler

#import other libraries for image manipulation

import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
import shutil
import seaborn as sns

#----------LOADING TRAINING DATA---------------
#Creating data generators
data = ImageDataGenerator(preprocessing_function=preprocess_input)
#training image set
train_generator = data.flow_from_directory(
    'C:/Users/mnyathi/Documents/nya-cnn/train',
    target_size = (227,227),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle=True)

#validation images set
val_generator =  data.flow_from_directory(
    'C:/Users/mnyathi/Documents/nya-cnn/val',
    target_size = (227,227),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle=True)

#Testing images set
test_generator =  data.flow_from_directory(
    'C:/Users/mnyathi/Documents/nya-cnn/test',
    target_size = (227,227),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle=False)
    
#----------BUILDING THE CUSTOM CNN MODEL---------------
#Creating the model using the sequential API in Tensorflow and naming it “Crackchecker”
Crackchecker = Sequential() 

#First convolutional block
Crackchecker.add(Conv2D(16,(3,3), activation = 'relu', padding='same', input_shape=(227, 227,3)))
Crackchecker.add(MaxPooling2D((2,2), padding='same'))

#Second convolutional block
Crackchecker.add(Conv2D(32,(3,3), activation = 'relu', padding='same'))
Crackchecker.add(MaxPooling2D((2,2), padding='same'))

#Third convolutional block
Crackchecker.add(Conv2D(64,(3,3), activation = 'relu', padding='same'))
Crackchecker.add(MaxPooling2D((2,2), padding='same'))

#Fourth convolutional block
Crackchecker.add(Conv2D(128,(3,3), activation = 'relu', padding='same'))
Crackchecker.add(MaxPooling2D((2,2), padding='same'))

#Global average pooling layer
Crackchecker.add(GlobalAveragePooling2D())

#Fully connected layer
Crackchecker.add(Dense(64, activation='relu'))
Crackchecker.add(BatchNormalization())

#Output layer
Crackchecker.add(Dense(2, activation = 'softmax'))

#Training options
#the training options are varied to create the different cases
learning_rate = 0.0001
optimizer = Adam(learning_rate = learning_rate)

#Compiling the model
Crackchecker.compile(optimizer= optimizer , loss='categorical_crossentropy', metrics=['accuracy'])

#Summary of the model
Crackchecker.summary() #printing a summary of the model

#--------Improving the performance of the model
#Early stopping to stop the model from overfitting by stopping training through callbacks
EarlyStop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience=15)

#This part of the code allows us to save the best model automatically
Checkpoint = ModelCheckpoint('C:/Users/mnyathi/Documents/nya-cnn/checkpoint', monitor = 'val_loss', mode='min', verbose=1, save_best_only= True)
cb_list=[EarlyStop, Checkpoint]

#--------TRAINING THE MODEL------------------
history = Crackchecker.fit(train_generator, epochs=100, validation_data = val_generator, callbacks = cb_list)
#The following section of the code shows the script used to plot the results training and testing results of the model. 
#--------PLOTTING THE TRAINING AND VALIDATION ACCURACY-----
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='lower right')
plt.savefig("accuracy_progress.jpg", dpi=300)

#--------PLOTTING THE TRAINING AND VALIDATION LOSS–----
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.savefig("loss_progress.jpg", dpi=300)

#---------TESTING THE MODEL----------------------------
# predicted labels for the test set
y_pred = np.argmax(Crackchecker.predict(test_generator), axis=1)

# true labels for the test set
y_true = test_generator.classes

# Computing the confusion matrix using predicted labels and true labels
cm = confusion_matrix(y_true, y_pred)
classes = ["Crack", "No Crack"]

# plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix.jpg", dpi=300)
plt.show()

# printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
