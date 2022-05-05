import os
import cv2 as cv
import numpy as np

def sharpen(img):
    # Defining a sharpening filter to shapen the image
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv.filter2D(img,-1,kernel)

def unsharp_masking(img):
    # Unsharp masking is an efficient technique to make the edges prominent while keeping the original distribution of pixel values
    gaussian_3 = cv.GaussianBlur(img, (0,0), 2.0)
    return cv.addWeighted(img, 2, gaussian_3, -1, 0)


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
    filepath1 = filename+"_cropped.JPG"
    cv.imwrite(filepath1, img)
    cv.namedWindow("Scale", cv.WINDOW_NORMAL)
    cv.imshow("Scale", img)
    
    # Sharpen the image so that the edges of the digits get emphasized which would be later used in contour detections
    img = unsharp_masking(img)
    img = sharpen(img)

    
    # Displaying the improved image after sharpening
    cv.imshow('Sharpened image', img) 
    cv.imwrite(filename+'-sharpened.JPG',img)

    cv.waitKey(0)
    cv.destroyAllWindows()
        
    
    

