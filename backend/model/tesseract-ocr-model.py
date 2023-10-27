# change directory to: "c:/users/wenji/desktop/nus academics/nus y3s1/dsa3101/dsa3101-2310-12-ocr/backend/model"

# pip install pytesseract numpy opencv-python imutils

# libraries for OCR
from PIL import Image
import pytesseract #also required for preprocessing
import numpy as np

# libraries for preprocessing 
from pytesseract import Output
import argparse
import imutils
import cv2

# Load image
## testing from local image
area = 'capt' #'capt' or 'rc4', 'u_town_residence', 'cinnamon'
time = 'after' #'after' or 'before'
phonetype = 'Android' #'iPhone' or 'Android'
photo = 'test-skewed' 
filetype = 'jpg' #'jpeg' or 'jpg'

filename = f'../../../12-ocr-image-data/{area}_{time}/{phonetype}/{photo}.{filetype}'

img1 = np.array(Image.open(filename))
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' 

# # Preprocessing
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required = True, help = "path to input image to be OCR'd")
# # args = vars(ap.parse_args())

# image = cv2.imread(filename)
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

# print("[INFO] detected orientation: {}".format(results["orientation"]))
# print("[INFO] rotate by {} degrees to correct". format(results["rotate"]))
# print("[INFO] detected script: {}".format(results["script"]))

# rotated = imutils.rotate_bound(image, angle=results["rotate"])

# cv2.imshow("Original", image)
# cv2.imshow("Output", rotated)
# cv2.waitKey(0)

# Preprocessing method 2: Deskewing
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
    print(angle)
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

image = cv2.imread(filename)
gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

cv2.imshow("Original", ResizeWithAspectRatio(image, height = 1000))
cv2.imshow("Output", ResizeWithAspectRatio(deskew(image), height=1000))
#cv2.imshow("Output", ResizeWithAspectRatio(gray, height=1000))

cv2.waitKey(0)

# # Convert image to text
text = pytesseract.image_to_string(deskew(cv2.imread(filename)))
print(text)