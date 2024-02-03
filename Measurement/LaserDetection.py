# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 23:31:16 2022

@author: mnyathi
"""
#----------IMPORTING LIBRARIES-----------------
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#----------PERFORMING CRACK DETECTION------------
# Importing the image
img = cv.imread('C:/users/mnyathi/downloads/circle.jpg')

# Converting from BGR to HSV colour space
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Defining lower and upper bounds for red colour
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Creating masks for both red ranges and then combining
mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
mask = cv.bitwise_or(mask1, mask2)

# morphological operation to reduce noise in the mask
kernel = np.ones((3, 3), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)

# Filtering the colour red from the original image using the mask
res = cv.bitwise_and(img, img, mask=mask)

# Conversion to grayscale to improve circle detection
gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

# Applying gaussiab blurto reduce noise and improve circle detection
gray_blurred = cv.GaussianBlur(gray, (9, 9), 2, 2)

# Applying Hough Circle detection. Adjust parameters to try and detect a single circle of correct size. try adjusting param1, param2 first
circles = cv.HoughCircles(gray_blurred, cv.HOUGH_GRADIENT, 1, gray_blurred.shape[0]/8, param1=100, param2=30, minRadius=0, maxRadius=0)

# Drawing the largest circle and displaying the diameter if any are detected
if circles is not None:
    circles = np.uint16(np.around(circles))
    largest_circle = max(circles[0, :], key=lambda x: x[2])
    cv.circle(img, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 5)
    cv.circle(img, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 5)
    
    # Calculating the diameter
    diameter = largest_circle[2] * 2
    print(f"Detected circle diameter: {diameter} pixels")
    
    # Putting text on the image showing the diameter
    cv.putText(img, f"Diameter: {diameter}px", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

    # Converting image to RGB for matplotlib display
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Displaying the image with matplotlib
    plt.imshow(img_rgb)
    plt.title('Detected Circle with Diameter')
    plt.axis('off')
    plt.show()
else:
    print("No circles were detected.")
