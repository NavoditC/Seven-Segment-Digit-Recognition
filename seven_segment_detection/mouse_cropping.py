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
        
    r = cv.selectROI("Select the area", img)

    # Crop image
    img = img[int(r[1]):int(r[1]+r[3]),
                      int(r[0]):int(r[0]+r[2])]

    # Display the scale with a slider for adjusting gamma
    filepath1 = filename+"_mouse_cropped.JPG"
    cv.imwrite(filepath1, img)
    cv.namedWindow("Scale", cv.WINDOW_NORMAL)
    cv.imshow("Scale", img)