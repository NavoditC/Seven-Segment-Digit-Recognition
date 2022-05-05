import os
import cv2 as cv
import numpy as np


if __name__ == "__main__":

    filename = input("Enter the filename: ")
    filepath = filename+".JPG"
    img = cv.imread(filepath)
   
    if img is None:
        print("Could not find or open the image")
        exit(0)
        

    # Manually crop the image
    img = img[1575:2075,1900:3300]

    # Display the scale with a slider for adjusting gamma
    filepath1 = filename+"_manual_cropped.JPG"
    cv.imwrite(filepath1, img)
    cv.namedWindow("Scale", cv.WINDOW_NORMAL)
    cv.imshow("Scale", img)