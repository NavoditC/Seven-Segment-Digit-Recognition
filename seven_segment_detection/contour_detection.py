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
        
    final = img.copy()
    img = cv.copyMakeBorder(img, 10, 10, 10, 10, borderType=cv.BORDER_CONSTANT, value=(255, 255, 255))
    # Convert image to gray-scale
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # Convert grayscale image to binary image
    thr,dst = cv.threshold(img_gray,60,255,cv.THRESH_BINARY)
    
    # Clean up the image
    for i in range(8):
         dst = cv.dilate(dst,None)
            
    #true_img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    lengths = np.zeros((len(contours),))

    for i,c in enumerate(contours):
         lengths[i] = cv.arcLength(c, True)
            
    threshold = np.quantile(lengths, 0.995)
    for i in range(len(contours)):
        contour_perimeter = cv.arcLength(contours[i], True)
        if contour_perimeter > threshold:
            cnts = contours[i]
            cv.drawContours(img, contours, i,(255,0,0), 5)

            
    img = img[10:img.shape[0]-10, 10:img.shape[1]-10]
    
    cv.imshow('Scale contour',img)
    cv.waitKey(0)
    cv.imwrite(filename+"_relevant_contour"+".JPG",img)

