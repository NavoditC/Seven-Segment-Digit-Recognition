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
    filepath1 = filename+"_cropped.JPG"
    cv.imwrite(filepath1, img)
    cv.namedWindow("Scale", cv.WINDOW_NORMAL)
    cv.imshow("Scale", img)


    def change(val):
        gamma = val / 10
        lut = np.empty((1, 256), np.uint8)
        for i in range(256):
            lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        corrected_img = cv.LUT(img, lut)
        cv.imshow("Scale", corrected_img)

    # Gamma is scaled up by a factor of 10 for the purpose of taskbar and then scaled down by a factor of 10
    cv.createTrackbar("Gamma", "Scale", 0, 100, change)
    cv.waitKey(0)

    gamma = int(cv.getTrackbarPos("Gamma", "Scale")) / 10
    print(f'The value of gamma used for gamma correction is {gamma}')

    lut= np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    improved_img = cv.LUT(img, lut)

    filepath2 = filename+"_improved.JPG"
    cv.imwrite(filepath2, improved_img)
    cv.destroyAllWindows()
