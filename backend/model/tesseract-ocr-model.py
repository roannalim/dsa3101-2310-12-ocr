from PIL import Image
import pytesseract
import numpy as np

#testing from local image
area = 'capt' #or 'rc4', 'u_town_residence', 'cinnamon'
time = 'after' #or 'before'
phonetype = 'iPhone' #or 'Android'
photo = 'IMG_6062' 

filename = f'../../../12-ocr-image-data/{area}_{time}/{phonetype}/{photo}.jpeg'

img1 = np.array(Image.open(filename))
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe' 

text = pytesseract.image_to_string(img1)
print(text)