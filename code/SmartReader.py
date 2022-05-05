# Import the necessary packages
from imutils import contours
import imutils
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import keyboard

# Define the dictionary of digit segments so we can identify each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# Search image files in test folder
folder_path = "test"
filename_list = os.listdir(folder_path)

xs = []
ys = np.zeros(len(filename_list))
cropping = False
x_1, y_1, x_2, y_2 = 0, 0, 0, 0


# Function to adjust the brightness of the image if there is a streak of sunlight across the display screen
def adjust_brightness(img):
    cv2.namedWindow("Scale", cv2.WINDOW_NORMAL)
    cv2.imshow("Scale", img)


    def change(val):
        gamma = val / 10
        lut = np.empty((1, 256), np.uint8)
        for i in range(256):
            lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        corrected_img = cv2.LUT(img, lut)
        cv2.imshow("Scale", corrected_img)

    # Gamma is scaled up by a factor of 10 for the purpose of taskbar and then scaled down by a factor of 10
    cv2.createTrackbar("Gamma", "Scale", 0, 100, change)
    cv2.waitKey(0)

    gamma = int(cv2.getTrackbarPos("Gamma", "Scale")) / 10
    print(f'The value of gamma used for gamma correction is {gamma}')

    lut= np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    improved_img = cv2.LUT(img, lut)
    cv2.destroyWindow("Scale")
    return improved_img

def sharpen_filter(img):
    # Defining a sharpening filter to shapen the image
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img,-1,kernel)

def unsharp_masking(img):
    # Unsharp masking is an efficient technique to make the edges prominent while keeping the original distribution of pixel values
    gaussian_3 = cv2.GaussianBlur(img, (0,0), 2.0)
    return cv2.addWeighted(img, 2, gaussian_3, -1, 0)

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_1, y_1, x_2, y_2, cropping
    # if the left mouse button was DOWN, 1 RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_1, y_1, x_2, y_2 = x, y, x, y
        cropping = True
    # Mouse is Moving
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_2, y_2 = x, y
    
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the second (x, y) coordinates
        x_2, y_2 = x, y
        cropping = False # cropping is finished
        refPoint = [(x_1, y_1), (x_2, y_2)]
        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            
def region_select(Image):
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_crop)
    print("Press 'E' key to stop cropping")
    while True:
        i = Image.copy()
        if not cropping:
            cv2.imshow("image", Image)
        elif cropping:
            cv2.rectangle(i, (x_1, y_1), (x_2, y_2), (255, 0, 0), 2)
            refPoints = [(x_1, y_1), (x_2, y_2)]
            cv2.imshow("image", i)
        cv2.waitKey(1)
        if keyboard.is_pressed('e'):  # if key 'E' is pressed 
            print('Selection made')
            break  # finishing the loop
    return refPoints, Image

def delete_dot(cropped_binary_image):
    img_delete_dot = cropped_binary_image.copy()
    # cv2.imshow("Operation", img_delete_dot)
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    while(1):
    # For the best result, 'e' three times then 'd' four times
        print("Press 'ESC' to escape.")
        print("Press 'E' or 'e' to to erode the image.")
        print("Press 'D' or 'd' to to dilate the image.")
        
        key_input = cv2.waitKey()
        print(key_input)
        if key_input == 27:
        # Press 'ESC' to escape
            cv2.imshow("Operation", img_delete_dot)
            break
        elif key_input == 69 or key_input == 101:
        # Press 'e' or 'E' to apply erode
            img_delete_dot = cv2.erode(img_delete_dot, ker)
            cv2.imshow("Operation", img_delete_dot)
        elif key_input == 68 or key_input == 100:
        # Press 'd' or 'D' to apply dilate
            img_delete_dot = cv2.dilate(img_delete_dot, ker)
            cv2.imshow("Operation", img_delete_dot)
    
    return img_delete_dot

def delete_dot_auto(cropped_binary_image, iteration_erosion, iteration_dilation):
    img_delete_dot_auto = cropped_binary_image.copy()
    # cv2.imshow("Operation", img_delete_dot_auto)
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    img_delete_dot_auto = cv2.erode(img_delete_dot_auto, ker, iterations = iteration_erosion)
    # cv2.imshow("Operation", img_delete_dot_auto)
    img_delete_dot_auto = cv2.dilate(img_delete_dot_auto, ker, iterations = iteration_dilation)
    # cv2.imshow("Operation", img_delete_dot_auto)
    
    return img_delete_dot_auto

# Write a new csv file to save data
header = ['index', 'filename', 'value']
with open('data/'+folder_path+'.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

#Select region by cropping the first image
img_filename = filename_list[0]

img = cv2.imread(folder_path+'/'+img_filename, cv2.IMREAD_COLOR)
img = imutils.resize(img, height=700)
oriImage = img.copy()
points, img = region_select(oriImage)

value = int(input('Enter 1 if the image requires its brightness to be adjusted else enter 0: '))
if value == 1:
    img_cropped = adjust_brightness(img)
     
# Start timer
start = time.process_time()

for idx_filename in range(len(filename_list)):
    img_filename = filename_list[idx_filename]
    img_original = cv2.imread(folder_path+'/'+img_filename, cv2.IMREAD_COLOR)
    
    # Preprocess the image by resizing it, converting it to graycale, blurring it, and computing an edge map
    img_resized = imutils.resize(img_original, height=700)
    img_cropped = img_resized[points[0][1]:points[1][1], points[0][0]:points[1][0]]
    # To convert the image to grayscale
    img_grayscale = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_sharpened = unsharp_masking(img_grayscale) # Unsharp masking to make the edges prominent while keeping the original distribution of pixel values
    img_sharpened = sharpen_filter(img_sharpened) # Use of a sharpening filter to further sharpen the image
    img_gaussianblur = cv2.GaussianBlur(img_sharpened, (3, 3), 0)
    # Apply Canny edge detection
    img_cannyedge = cv2.Canny(img_gaussianblur, 50, 200, 255)
    # To convert the image to binary
    (thr, dst) = cv2.threshold(img_grayscale, 128, 255, cv2.THRESH_BINARY)
    img_binary = dst.copy()
    img_operation = img_binary.copy()
    # Apply function of deleting dotsbo
    iteration_erosion = 3
    iteration_dilation = 4
    img_delete_dot = delete_dot_auto(img_operation, iteration_erosion, iteration_dilation)
    # Threshold the cropped image, then apply a series of morphological operations to cleanup the thresholded image
    thresh = cv2.threshold(img_grayscale, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = 255-cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the thresholded image, then initialize the digit contours lists
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []
    
    # Loop over the digit area candidates
    for c in cnts:
        # Compute the bounding box of the contours
        (x, y, w, h) = cv2.boundingRect(c)
        # print('top')
        # print(x, y, w, h) 
        # If the contour is sufficiently large, it must be a digit
        if h > 40:
            if w < 15:
                min_pixel_width = 26
                x = x + w - min_pixel_width
                w = min_pixel_width
            digitCnts.append(c)
        # print('top after')
        # print(x, y, w, h)
            
    # Sort the contours from left-to-right, then initialize the actual digits themselves
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    digits = []
    expected_digits = 6
    
    img_bounding = img_cropped.copy()
    
    if len(digitCnts) == expected_digits:
        # Loop over each of the digits
        for c in digitCnts:
            # Extract the digit ROI
            (x, y, w, h) = cv2.boundingRect(c)
            # print('bottom')
            # print(x, y, w, h)
            if h > 40:
                if w < 15:
                    min_pixel_width = 26
                    x = x + w - min_pixel_width
                    w = min_pixel_width
            # print('bottom after')
            # print(x, y, w, h)
            roi = thresh[y:y + h, x:x + w]
            # Compute the width and height of each of the 7 segments we are going to examine
            (roiH, roiW) = roi.shape
            # (dW, dH) = (int(roiW * 0.35), int(roiH * 0.15))
            (dW, dH) = (int(roiW * 0.3), int(roiH * 0.13))
            dHC = int(roiH * 0.08)
            # dHC = int(roiH * 0.11)
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
            image = cv2.rectangle(img_bounding, (x,y), (x+w,y+h), (255,0,0), 5)
            
            # Loop over the segments   
            for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                # Extract the segment ROI, count the total number of
                # thresholded pixels in the segment, and then compute
                # the area of the segment
                segROI = roi[yA:yB, xA:xB]
                total = cv2.countNonZero(segROI)
                area = (xB - xA) * (yB - yA)
                
                # If the total number of non-zero pixels is greater than
                # 50% of the area, mark the segment as "on"
                if total / float(area) > 0.45:
                    on[i]= 1
            
            # print(on)
            if on == [1, 0, 1, 1, 1, 0, 0]:
                # Fake 2 should be converted to 2.
                on = [1, 0, 1, 1, 1, 0, 1]
            if on == [0, 0, 1, 0, 0, 0, 0]:
                # Fake 1 should be converted to 1.
                on = [0, 0, 1, 0, 0, 1, 0]
            if on == [1, 1, 1, 1, 0, 0, 1]:
                # Fake 9 should be converted to 9.
                on = [1, 1, 1, 1, 0, 1, 1]
            if on == [1, 0, 1, 0, 0, 0, 0]:
                # Fake 7
                on = [1, 0, 1, 0, 0, 1, 0]
            if on == [1, 1, 1, 0, 1, 1, 1] or \
               on == [0, 0, 1, 0, 0, 1, 0] or \
               on == [1, 0, 1, 1, 1, 0, 1] or \
               on == [1, 0, 1, 1, 0, 1, 1] or \
               on == [0, 1, 1, 1, 0, 1, 0] or \
               on == [1, 1, 0, 1, 0, 1, 1] or \
               on == [1, 1, 0, 1, 1, 1, 1] or \
               on == [1, 0, 1, 0, 0, 1, 0] or \
               on == [1, 1, 1, 1, 1, 1, 1] or \
               on == [1, 1, 1, 1, 0, 1, 1]:
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
                # print(on)
                # print('good')
            # else:
            #     print(on)
            #     print(img_filename)
            #     print(len(digits))
            
        if len(digits) == expected_digits:
            value = float(0)
            for idx in range(len(digits)):
                value += float(digits[idx])*(10**(1-idx))
                value = np.round(value, decimals = 4)
        else:
            value = float(-1)
            
    else:
        value = float(-1)
    
    print(value)
    
    # Write a cvs file
    with open('data/'+folder_path+'.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([idx_filename, img_filename, value])
        
    # Plot datapoint
    xs = np.linspace(0,len(filename_list), len(filename_list))
    ys[idx_filename] = value
    
    plt.cla()
    plt.plot(xs, ys)
    plt.title('Data')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.pause(0.05)
    plt.show(block=False)
      
#Calculate the time processing has taken
print("{} seconds".format(time.process_time() - start))
    
# Read the images
# cv2.imshow("Resized", img_resized)
# cv2.imshow("Cropped", img_cropped)    
# cv2.imshow("Bounding box", image)
cv2.waitKey()     
    
# close all open windows
cv2.destroyAllWindows()
