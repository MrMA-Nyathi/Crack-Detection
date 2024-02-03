# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:21:16 2023

@author: mnyathi
"""
#----------IMPORTING NECESSARY LIBRARIES---------------------------------

import numpy as np
from skimage import morphology
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from PIL import Image
import os

#---------Changing the working directory
os.chdir("C:/Users/mnyathi/Documents/segmentation/new images/lengths")

#-----------CREATING A CRACK WIDTH MEASUREMENT FUNCTION------------------------

def crack_width_measure(binary_image, display_results=True):
    """
    Measure the crack widths along its length and identify the maximum crack width in a binary image.
    
    :param binary_image: A 2D NumPy array representing the binary image (where the crack is represented as by white pixels).
    :param display_results: If True, display the results.
    :return: A tuple containing an array of crack widths and the maximum crack width.
    """
    # Ensuring the image being used is a binary image
    binary_image = binary_image > 0
    
    # Applying skeletonisation
    skeleton = morphology.skeletonize(binary_image)
    
    # Applying distance transform
    distance_transform = distance_transform_edt(binary_image)
    
    # Measuring crack widths along the skeleton
    crack_widths = distance_transform[skeleton]*2
    
    # Identifying maximum crack width in the image
    max_crack_width = np.max(crack_widths)
    
    
    if display_results:
        # Display the original binary image
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_image, cmap='gray')
        plt.title('Binary Image')
        plt.axis('off')
        plt.show()
        
        # Display the skeleton
        plt.figure(figsize=(6, 6))
        plt.imshow(skeleton, cmap='gray')
        plt.title('Skeleton')
        plt.axis('off')
        plt.show()
        
        # Display the distance transform
        plt.figure(figsize=(6, 6))
        plt.imshow(distance_transform, cmap='gray')
        plt.title('Distance Transform')
        plt.axis('off')
        plt.show()
        
        # Display the crack widths along the skeleton
        plt.figure(figsize=(6, 6))
        plt.imshow(binary_image, cmap='gray')
        plt.scatter(*np.where(skeleton)[::-1], c=crack_widths, cmap='jet', s=10, label='Crack Widths')
        plt.colorbar(label='Crack Width (pixels)')
        plt.title('Detected crack')
        plt.axis('off')
        
        # Print value of max crack width
        print(f"Maximum Crack Width: {max_crack_width} pixels")
        
        text_x = 10  # 10 pixels from the left
        text_y = 20  # 20 pixels from the top
        
        plt.text(text_x, text_y, f"Max Width: {max_crack_width:.2f} pixels", color='white')
        
        plt.savefig("crackwidth.jpg", dpi=300)
        #plt.legend()
        plt.show()
        
        
    
    return crack_widths, max_crack_width

#CREATING A FUNCTION TO ENSURE CORRECT IMAGE TYPE IS LOADED INTO MEASURING FUNCTION at all times-------------------

def analyse_crack_image(image_path):
    """
    Load an image from a file path, convert it to a binary format, and analyze the crack widths.
    
    :param image_path: Path to the image file.
    :return: A tuple containing an array of crack widths and the maximum crack width.
    """
    # Loading the image
    image = Image.open(image_path)

    # Converting the image to a NumPy array
    image_array = np.array(image)

    # Ensuring the image is in binary format
    # Here, we assume the crack is represented by white pixels.
    binary_image = image_array > 0

    # Analysing the crack widths
    crack_widths, max_crack_width =crack_width_measure(binary_image)
    
    return crack_widths, max_crack_width

#USING THE FUNCTION:
crack_widths, crack_width_measure = analyse_crack_image('C:/Users/mnyathi/Documents/segmentation/imageG135.png')
