# pip install torch torchvision torchaudio easyocr numpy opencv-python matplotlib pillow

# libraries for OCR
import cv2
import easyocr
import numpy as np
import queue
import re
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from io import BytesIO
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

local_image = True

class MyDataset(IterableDataset):
    def __init__(self, image_queue, transforms = None):
      self.queue = image_queue
      self.transforms = transforms

    def read_next_image(self):
        while self.queue.qsize() > 0:
            if (self.transforms is not None):
                self = self.transforms(self) 
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()

def pytorch_easy_ocr(image, debugging = False):

    # load trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    num_classes = 2 # 1 class (desired box stating gross weight value) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load('./best_trained_object_detection_model.pth'))
    model.eval()
    
    # load input image
    buffer = queue.Queue()
    if type(image) is str:
        new_input = Image.open(image)
    else:
        new_input = Image.open(BytesIO(image))
    buffer.put(TF.to_tensor(new_input)) 

    # load image into dataloader
    dataset = MyDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=1)

    outputs = None

    for data in dataloader:
        images = list(image.to(torch.device("cpu")) for image in data)
        outputs = model(data)

    boxes = outputs[0]['boxes'].data.cpu().numpy()    # Format of the output's box is [Xmin,Ymin,Xmax,Ymax]
    scores = outputs[0]['scores'].data.cpu().numpy()
    
    if (boxes is not None):
        box = boxes[0].astype(np.int32) # select only those boxes with highest score
    else:
        return ""
        
    if (scores is not None):                      
        score = scores[0]                # select only those scores with highest score 
        print("Bounding Box Detection Score: ", score)

    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    area = (xmin, ymin, xmax, ymax)
    
    # Filter image to bounding box
    region = new_input.crop(area)
    opencvImage = cv2.cvtColor(np.array(region), cv2.COLOR_BGR2GRAY)

    # Histogram equalisation
    equ = cv2.equalizeHist(opencvImage)

    # Manual thresholding
    th2 = 65 # Vary this threshold to determine extent of Histogram Equalisation
    equ[equ>=th2] = 255
    equ[equ<th2]  = 0

    if local_image:
        photo = re.search(r'[\/][^\/.]+[.]', filename).group()[1:-1]
        filetype = re.search(r'.[j][p].+', filename).group()
        right = "/"+photo+filetype
        path = filename.partition(right)[0]
        cv2.imwrite(f'{path}/{photo}_cropped.jpg', equ)

    # Initialise EasyOCR Reader
    reader = easyocr.Reader(['en'])
    
    # Obtain OCR Result
    ocr_result = reader.readtext(equ, # region of interest image
                                 allowlist = '0123456789', # only output digits
                                 detail = 1, # less-detailed and simple output
                                 paragraph = False, # no paragraphing in image
                                 rotation_info = [0, 90, 180, 270], # try all possible text orientations
                                 width_ths = 0.2) #maximum horizontal distance for bounding box merging
    final_result = ""
    print(ocr_result)

    if (ocr_result != []):
        highest_score = 0
        desired_ocr_result = None
        for i in range(len(ocr_result)):
            if ocr_result[i][2] > highest_score:
                desired_ocr_result = ocr_result[i]
                highest_score = ocr_result[i][2]

        print("EasyOCR Results:\n",
            "Recognised Text = ", desired_ocr_result[1], "\n",
            "Text Recognition Score = ", desired_ocr_result[2])
        
        final_result = desired_ocr_result[1]
    else: 
        print("No digits recognised")

    if debugging: # Show image with bounding boxes

        sample = images[0].permute(1,2,0).cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        sample = np.ascontiguousarray(sample)

        cv2.rectangle(sample,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (220, 0, 0), 2)
            
        ax1.set_axis_off()
        ax1.imshow(sample)
        ax1.set_title("Original Image with\nObject Detection Bounding Box")

        if (ocr_result != []):
            bbox = desired_ocr_result[0]

            print(bbox)
            equ = np.ascontiguousarray(equ)

            start_point = (int(bbox[0][0]), int(bbox[1][1]))
            end_point = (int(bbox[2][0]), int(bbox[3][1]))

            cv2.rectangle(equ,
                        start_point,
                        end_point,
                        (220, 0, 0), 2)                        

        ax2.set_axis_off()
        ax2.imshow(equ)
        ax2.set_title("Cropped Image with\nEasyOCR Bounding Box")

        plt.show(block = True)

    return final_result

if local_image:
    ## testing from local image
    area = 'cinnamon' #'capt' or 'rc4', 'u_town_residence', 'cinnamon'
    time = 'after' #'after' or 'before'
    phonetype = 'Android' #'iPhone' or 'Android'
    photo = '20231006_084748' #insert photo file name without extension
    filetype = 'jpg' #'jpeg' or 'jpg'

    filename = f'../../../../12-ocr-image-data/{area}_{time}/{phonetype}/{photo}.{filetype}'
    print(pytorch_easy_ocr(filename, debugging = True))