# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 13:43:33 2023

@author: mnyathi
"""
#-------------------IMPORTING LIBRARIES--------------------------------------
import os
os.chdir("C:/Users/mnyathi/Documents/Segmentation/segv3")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Concatenate, BatchNormalization, ReLU, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from pycocotools.coco import COCO
import cv2 as cv
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import Reshape
from sklearn.metrics import precision_recall_curve

#------------------------------LOADING DATASETS-------------------------------------

#TRAINING DATA------------------------------------------
annotation_data = "C:/Users/mnyathi/Documents/Segmentation/data/train/_annotations.coco.json"
coco = COCO(annotation_data)

#loading IDs for all images in the coco file
image_ids = coco.getImgIds()

X_train = []
Y_train = []
#loading data for each image
for image_id in image_ids:
    image =coco.loadImgs(image_id)[0]
    image_path = "C:/Users/mnyathi/Documents/Segmentation/data/train/" + image["file_name"]
    image_data = plt.imread(image_path)
    resized_image = cv.resize(image_data, (160, 160))

    # Loading the annotation data
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    train_annotations = coco.loadAnns(annotation_ids)

    #processing the annotations
    for train_annotation in train_annotations:
        binary_mask = coco.annToMask(train_annotation)
        binary_mask = cv.resize(binary_mask, (160,160))

        X_train.append(resized_image)
        Y_train.append(binary_mask)
    
#converting training data to numpy
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#checking if the training data was created correctly
plt.imshow(X_train[1])
plt.show()
plt.imshow(Y_train[1])
plt.show()
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)

#VALIDATION DATA------------------------------------------
annotation_data = "C:/Users/mnyathi/Documents/Segmentation/data/valid/_annotations.coco.json"
coco = COCO(annotation_data)

#loading IDs for allimages in the coco file
image_ids = coco.getImgIds()

X_val = []
Y_val = []
#loading data for each image
for image_id in image_ids:
    image =coco.loadImgs(image_id)[0]
    image_path = "C:/Users/mnyathi/Documents/Segmentation/data/valid/" + image["file_name"]
    image_data = plt.imread(image_path)
    resized_image = cv.resize(image_data, (160, 160))

    # Loading the annotation data
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    train_annotations = coco.loadAnns(annotation_ids)

    #processing the annotations
    for train_annotation in train_annotations:
        binary_mask = coco.annToMask(train_annotation)
        binary_mask = cv.resize(binary_mask, (160,160))

        X_val.append(resized_image)
        Y_val.append(binary_mask)
    
#converting training data to numpy
X_val = np.array(X_val)
Y_val = np.array(Y_val)

#checking if the validation data was created correctly
plt.imshow(X_val[4])
plt.show()
plt.imshow(Y_val[4])
plt.show()
print("X_val shape:", X_val.shape)
print("Y_val shape:", Y_val.shape)

#TESTING DATA------------------------------------------
annotation_data = "C:/Users/mnyathi/Documents/Segmentation/data/test/_annotations.coco.json"
coco = COCO(annotation_data)

#loading IDs for all images in the coco file
image_ids = coco.getImgIds()

X_test = []
Y_test = []
#loading data for each image
for image_id in image_ids:
    image =coco.loadImgs(image_id)[0]
    image_path = "C:/Users/mnyathi/Documents/Segmentation/data/test/" + image["file_name"]
    image_data = plt.imread(image_path)
    resized_image = cv.resize(image_data, (160, 160))

    # Loading the annotation data
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    train_annotations = coco.loadAnns(annotation_ids)

    #processing the annotations
    for train_annotation in train_annotations:
        binary_mask = coco.annToMask(train_annotation)
        binary_mask = cv.resize(binary_mask, (160,160))

        X_test.append(resized_image)
        Y_test.append(binary_mask)
    
#converting training data to numpy
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#checking if the validation data was created correctly
plt.imshow(X_test[1])
plt.show()
plt.imshow(Y_test[1])
plt.show()
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

#------------------BUILDING THE NETWORK--------------------------------

inputs = Input(shape=(160,160,3))

#ENCODER BLOCK

#layer1
#first convolution
conv1 = Conv2D(64, 5, padding = 'same', activation = 'relu')(inputs)
batch_norm1=BatchNormalization()(conv1)
#second convolution
conv2= Conv2D(64, 5, padding ="same", activation = 'relu')(batch_norm1)
batch_norm2=BatchNormalization()(conv2)
MaxPool1 = MaxPooling2D(pool_size=(2,2))(batch_norm2)
skip1 = conv2

#layer 2
#first convolution
conv3 = Conv2D(128, 3, padding = 'same', activation = 'relu')(MaxPool1)
batch_norm3=BatchNormalization()(conv3)
#second convolution
conv4= Conv2D(128, 3, padding ="same", activation = 'relu')(batch_norm3)
batch_norm4=BatchNormalization()(conv4)
MaxPool2 = MaxPooling2D(pool_size=(2,2))(batch_norm4)
skip2 = conv4

#layer 3
#first convolution
conv5 = Conv2D(256, 3, padding = 'same', activation = 'relu')(MaxPool2)
batch_norm5=BatchNormalization()(conv5)
#second convolution
conv6= Conv2D(256, 3, padding ="same", activation = 'relu')(batch_norm5)
batch_norm6=BatchNormalization()(conv6)
MaxPool3 = MaxPooling2D(pool_size=(2,2))(batch_norm6)
skip3 = conv6

#layer 4
#first convolution
conv7= Conv2D(512, 3, padding = 'same', activation = 'relu')(MaxPool3)
batch_norm7=BatchNormalization()(conv7)
#second convolution
conv8= Conv2D(512, 3, padding ="same", activation = 'relu')(batch_norm7)
batch_norm8=BatchNormalization()(conv8)
MaxPool4 = MaxPooling2D(pool_size=(2,2))(batch_norm8)
skip4 = conv8

#mid convolution block
conv9 =  Conv2D(1024, 3, padding ="same")(MaxPool4)

#DECODER BLOCK
#decoding
up1 = Conv2DTranspose(512, (2,2), strides = 2, padding ="same")(conv9)
connect_skip4 = Concatenate()([up1,skip4])
conv10 = Conv2D(512, 5, padding="same", activation = 'relu')(connect_skip4)
batch_norm10=BatchNormalization()(conv10)
conv11 = Conv2D(512, 3, padding="same", activation = 'relu')(batch_norm10)

up2 = Conv2DTranspose(256, (2,2), strides = 2, padding ="same")(conv11)
connect_skip3 = Concatenate()([up2,skip3])
conv12 = Conv2D(256, 5, padding="same", activation = 'relu')(connect_skip3)
batch_norm11=BatchNormalization()(conv12)
conv13 = Conv2D(256, 3, padding="same", activation = 'relu')(batch_norm11)

up3 = Conv2DTranspose(128, (2,2), strides = 2, padding ="same")(conv13)
connect_skip2 = Concatenate()([up3,skip2])                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
conv14 = Conv2D(128, 5, padding="same", activation = 'relu')(connect_skip2)
batch_norm12=BatchNormalization()(conv14)
conv15 = Conv2D(128, 3, padding="same", activation = 'relu')(batch_norm12)

up4 = Conv2DTranspose(64, (2,2), strides = 2, padding ="same")(conv15)
connect_skip1 = Concatenate()([up4,skip1])
conv16 = Conv2D(64, 5, padding="same", activation = 'relu')(connect_skip1)
batch_norm13=BatchNormalization()(conv16)
conv17 = Conv2D(64, 3, padding="same", activation = 'relu')(batch_norm13)

tf.keras.layers.Flatten()

conv_final = Conv2D(1, 1, activation='sigmoid', padding='same')(conv17)

model = Model(inputs=inputs, outputs=conv_final)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=False, show_dtype=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

#----------------------TRAINING THE MODEL------------------------------------

checkpointer = tf.keras.callbacks.ModelCheckpoint('C:/Users/mnyathi/Documents/Segmentation/best_model.h5', monitor = 'val_loss', mode='min', verbose=1, save_best_only=True)
callbacks = [checkpointer,
        tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', mode = 'min'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, epochs=150, validation_data=(X_val, Y_val), verbose=1, callbacks = callbacks)
#------------training and validation accuracy and loss plots-----------------
# Plotting the training & validation accuracy plots
plt.plot(np.array(results.history['accuracy']) * 100)
plt.plot(np.array(results.history['val_accuracy']) * 100)
plt.title('Model Accuracy')
plt.ylabel('Accuracy (%)')
plt.xlabel('Epoch')
plt.ylim([0, 100])  
plt.legend(['Train', 'Validation'], loc='lower right')
plt.savefig("accuracy.jpg", dpi=300)

# Plotting the training & validation loss plots
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("loss_progress.jpg", dpi=300)
#---------------------RESULTS-------------------------------------------------
best_model = tf.keras.models.load_model("C:/Users/mnyathi/Documents/Segmentation/best_model.h5")

#Testing the model----
test_loss, test_accuracy = best_model.evaluate(X_test, Y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)

#---------------------Predicting using the model------------------------------
# Getting the model predictions on the validation set
y_pred_val = best_model.predict(X_val)

# Computing the precision-recall curve
precisions, recalls, thresholds = precision_recall_curve(Y_val.flatten(), y_pred_val.flatten())

# Computing the F1 scores for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)

# Getting the threshold that gives the highest F1 score
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Predicting using the saved best model on the test set
predicted_masks = best_model.predict(X_test)

# Applying optimal threshold on predicted masks
predicted_masks[predicted_masks >= optimal_threshold] = 1
predicted_masks[predicted_masks < optimal_threshold] = 0

# Saving visual representations of the predicted masks
save_dir = "C:/Users/mnyathi/Documents/Segmentation/segv3"

# Looping through all test images
for i in range(len(X_test)):
    
    # Predicting the mask for the current image
    results = best_model.predict(np.expand_dims(X_test[i], axis=0))

    # Converting raw prediction to a binary mask
    threshold = 0.5
    binary_mask = (results[0] > threshold).astype(np.uint8) * 255
    binary_mask = np.squeeze(binary_mask)

    # Creating a colored mask to identify the mask more clearer
    colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
    colored_mask[binary_mask == 255] = [0, 255, 0]  # green mask

    # Overlaying the mask onto the original image
    overlay = cv.addWeighted(X_test[i], 1, colored_mask, 0.5, 0)

    # Displaying the original image, ground truth, predicted binary mask, and overlay in one image
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(X_test[i])
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(Y_test[i], cmap='gray')  
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Predicted Binary Mask')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('Overlayed Image')
    axes[3].axis('off')

    # Saving the figure as a JPG image
    fig_name = os.path.join(save_dir, f"result_{i}.jpg")
    plt.savefig(fig_name, dpi=300)
    plt.show()
#-------------------Evaluation metrices---------------------------------------
def global_metrics(y_true, y_pred):
    # Ensuring the masks are in binary format
    y_pred = np.round(y_pred)
    y_true = np.round(y_true)

    # Flattening the arrays
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    # Calculating true positives, false positives, true negatives, false negatives
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    # IoU
    iou = tp / (tp + fp + fn)

    # Recall
    recall = tp / (tp + fn)

    # Precision
    precision = tp / (tp + fp)

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)

    return iou, recall, precision, f1

# Function for calculating global metrics
iou, recall, precision, f1 = global_metrics(Y_test, predicted_masks)

print(f"Global IoU: {iou}")
print(f"Global Recall: {recall}")
print(f"Global Precision: {precision}")
print(f"Global F1 Score: {f1}")
