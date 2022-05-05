import os
import imutils
from imutils import contours
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
    img_cropped = img[int(r[1]):int(r[1]+r[3]),
                      int(r[0]):int(r[0]+r[2])]
    img_grayscale = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)
    img_sharpened = cv.GaussianBlur(img_grayscale, (5, 5), 0)
    # Apply Canny edge detection
    img_cannyedge = cv.Canny(img_sharpened, 50, 200, 255)
    
    # Threshold the cropped image, then apply a series of morphological operations to cleanup the thresholded image
    thresh = cv.threshold(img_grayscale, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 5))
    thresh = 255-cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    
    # Find contours in the thresholded image, then initialize the digit contours lists
    cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    
    # Loop over the digit area candidates
    for c in cnts:
        # Compute the bounding box of the contours
        (x, y, w, h) = cv.boundingRect(c)
        # If the contour is sufficiently large, it must be a digit
        if h > 55:
            digitCnts.append(c)
            
    # Sort the contours from left-to-right, then initialize the actual digits themselves
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []
    
    img_bounding = img_cropped.copy()
    if len(digitCnts) == 6:
        # Loop over each of the digits
        for c in digitCnts:
            # Extract the digit ROI
            (x, y, w, h) = cv.boundingRect(c)
            if w < 27:
                min_pixel_width = 32
                x = x + w - min_pixel_width
                w = min_pixel_width
            roi = thresh[y:y + h, x:x + w]
            # Compute the width and height of each of the 7 segments we are going to examine
            (roiH, roiW) = roi.shape
            (dW, dH) = (int(roiW * 0.3), int(roiH * 0.15))
            dHC = int(roiH * 0.1)
            # Define the set of 7 segments
            segments = [
                ((0, 0), (w, dH)),	# top
                ((0, 0), (dW, h // 2)),	# top-left
                ((w - dW, 0), (w, h // 2)),	# top-right
                ((0, (h // 2) - dHC) , (w, (h // 2) + dHC)), # center
                ((0, h // 2), (dW, h)),	# bottom-left
                ((w - dW, h // 2), (w, h)),	# bottom-right
                ((0, h - dH), (w, h))	# bottom
            ]
            on = [0] * len(segments)
            image = cv.rectangle(img_bounding, (x,y), (x+w,y+h), (255,0,0), 5)
            
    cv.imshow("Bounding box", image)
    
    cv.imwrite(filename+'_cropped.JPG',img_cropped)
    cv.imwrite(filename+'_bounding_boxes.JPG',img_bounding)
    cv.waitKey(0)
    cv.destroyAllWindows()