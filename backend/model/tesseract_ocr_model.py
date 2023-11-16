# pip install pytesseract numpy opencv-python imutils

# libraries for OCR
from PIL import Image
import pytesseract #also required for preprocessing
import numpy as np
import re

# libraries for preprocessing 
from pytesseract import Output
import argparse
import imutils
import cv2

local_image = False # set to True if want to debug with image on local machine

def tesseract_ocr(filename, debugging = False):
    #if want to show deskew process, set debugging = True
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' 

    # Preprocessing functions: Deskewing
    def getSkewAngle(cvImage) -> float:
        newImage = cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations = 5)

        contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)

        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        angle = minAreaRect[-1]

        if (angle > 45):
            angle = angle - 90
        return angle

    def rotateImage(cvImage, angle: float):
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(newImage, M, (w, h), flags = cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return newImage

    def deskew(cvImage):
        angle = getSkewAngle(cvImage)
        return rotateImage(cvImage, angle)

    image = cv2.imread(filename)

    if debugging:
        #resize for cv2.imshow() on local machine
        def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
            dim = None
            (h, w) = image.shape[:2]

            if width is None and height is None:
                return image
            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
            else:
                r = width / float(w)
                dim = (width, int(h * r))

            return cv2.resize(image, dim, interpolation=inter)
        
        cv2.imshow("Original", ResizeWithAspectRatio(image, height = 1000))
        cv2.imshow("Deskewed", ResizeWithAspectRatio(deskew(image), height=1000))

        cv2.waitKey(0)

    # Convert image to text
    text = pytesseract.image_to_string(deskew(image))

    if text is not None: #no text recognised in image
        weight_4digit = re.match(r'\d{4}', text)
        if weight_4digit is None: #no 4 digit numbers matched
            weight_3digit = re.match(r'\d{3}', text)
            if weight_3digit is None: #no 3 digit numbers matched
                weight_2digit = re.match(r'\d{2}', text)
                if weight_2digit is None: #no 2 digit numbers matched
                    return ""
                else:
                    return weight_2digit.group(0)
            else:
                return weight_3digit.group(0)
        else:
            return weight_4digit.group(0)
    else:
        return ""

if local_image:
    ## testing from local image
    area = 'capt' #'capt' or 'rc4', 'u_town_residence', 'cinnamon'
    time = 'after' #'after' or 'before'
    phonetype = 'Android' #'iPhone' or 'Android'
    photo = '20231006_085648' #insert photo file name without extension
    filetype = 'jpg' #'jpeg' or 'jpg'

    filename = f'../../../12-ocr-image-data/{area}_{time}/{phonetype}/{photo}.{filetype}'
    print(tesseract_ocr(filename))