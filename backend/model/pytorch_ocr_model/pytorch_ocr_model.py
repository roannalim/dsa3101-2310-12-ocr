# Total 11 Checkpoints
# Checkpoints 4 to 6 only printed if train_model_flag == True

import numpy as np
import pandas as pd

import os

import cv2
import re
import torch
import torchvision
import easyocr
from torchvision.transforms import v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from matplotlib import pyplot as plt


cpu = torch.device("cpu") # Code not optimised for CUDA GPU usage

show_train_image_flag = False # set to True if want to show plot of training image with bounding box
show_test_image_flag = True # set to True if want to show plot of training image with bounding box
train_model_flag = False # set to True if want to re-train model

train_df = pd.read_csv("train_data.csv")

print("Checkpoint 1: Imports successful + CSV files successfully read")

train_transform = v2.Compose([v2.ToImage(), 
                              v2.ToDtype(torch.float32, scale=True)])

# Custom Dataset Class

class ScreenDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms = None, train = True):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.train = train

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if self.transforms is not None:  # Apply specified transformations
            image = self.transforms(image)

        if (self.train == False): # For test data
            return image, image_id
        
        # For train data
        records = self.df[self.df['image_id'] == image_id]   
        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        return image, target, image_id 
    
train_dir = '../../../../12-ocr-image-data-ocr-training/train' # to be adjusted by user accordingly

class Averager: # Returns the average loss
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
def collate_fn(batch):
    return tuple(zip(*batch))

print("Checkpoint 2: Custom classes defined")

train_dataset = ScreenDataset(train_df, train_dir, train_transform, True)

train_data_loader = DataLoader(
    train_dataset,
    batch_size = 10,
    shuffle = True,
    num_workers = 0,
    collate_fn = collate_fn
)

print("Checkpoint 3: Train DataLoader successfully initialised")

if (__name__ == '__main__') & (show_train_image_flag == True):
    images, targets, image_ids = next(iter(train_data_loader))
    print(image_ids[4])
    images = list(image.to(cpu) for image in images)
    targets = [{k: v.to(cpu) for k, v in t.items()} for t in targets]

    boxes = targets[4]['boxes'].cpu().numpy().astype(np.int32)
    sample = images[4].permute(1,2,0).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    sample = np.ascontiguousarray(sample)

    for box in boxes:
        cv2.rectangle(sample,
                    (box[0], box[1]),
                    (box[2], box[3]),
                    (220, 0, 0), 3)
        
    ax.set_axis_off()
    ax.imshow(sample)
    plt.show(block = True)

if (train_model_flag == True):
    print("Checkpoint 4: Entering Model Training Phase")
    pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    torch.save(pretrained_model.state_dict(), './pretrained_fasterrcnn_model_weights.pth')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    model.load_state_dict(torch.load('./pretrained_fasterrcnn_model_weights.pth'), strict = False)

    print("Checkpoint 5: Model successfully loaded")

    num_classes = 2 # 1 class (desired box stating gross weight value) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # training model
    model.train()

    model.to(cpu)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    num_epochs = 10

    loss_hist = Averager()
    itr = 1

    print("Checkpoint 6: Iterating through epochs")

    for epoch in range(num_epochs):
        loss_hist.reset()
        
        for images, targets, image_ids in train_data_loader:
            
            images = list(image.to(cpu) for image in images)
            targets = [{k: v.to(cpu) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)   # Return the loss

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)  # Average out the loss

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")

        least_loss = 1

        if (loss_hist.value < least_loss):
            torch.save(model.state_dict(), './best_retrained_object_detection_model.pth')

print("Checkpoint 7: Model Testing Phase")

# Load retrained model

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

num_classes = 2 # 1 class (desired box stating gross weight value) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('./best_retrained_object_detection_model.pth'))

test_transform = v2.Compose([v2.ToImage(), 
                             v2.ToDtype(torch.float32, scale=True),
                             v2.Resize((4000, 3000), antialias = True)])

test_df = pd.read_csv("test_data.csv")
test_dir = '../../../../12-ocr-image-data-ocr-training/test' # to be adjusted by user accordingly

test_dataset = ScreenDataset(test_df, test_dir, test_transform, False)
test_data_loader = DataLoader(
    test_dataset,
    batch_size = 10,
    shuffle = False
)

print("Checkpoint 8: Test DataLoader successfully initialised")

detection_threshold = 0.1 #Set threshold value for predicting bounding box

def format_prediction_string(box, score): # Define function for storing prediction results
    pred_string = "{0:.4f} {1} {2} {3} {4}".format(score, box[0], box[1], box[2], box[3])
    return pred_string

print("Checkpoint 9: Model Prediction Phase")
wanted = 0
results=[]
model.eval()

for images, image_ids in test_data_loader:    

    images = list(image.to(cpu) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()    # Format of the output's box is [Xmin,Ymin,Xmax,Ymax]
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        if (boxes is not None):
            box = boxes[0].astype(np.int32) # select only those boxes with highest score
            
        if (scores is not None):                      
            score = scores[0]                # select only those scores with highest score   

        image_id = image_ids[i]
        if (image_id == "20231006_085648.jpg"):
            wanted = i

        result = {                                     # Store the image id and boxes and scores in result dict.
            'image_id': image_id,
            'PredictionString': format_prediction_string(box, score),
            'score': score,
            'box_xmin': box[0],
            'box_ymin': box[1],
            'box_xmax': box[2],
            'box_ymax': box[3]
        }

        results.append(result) # Append result dict to Results list

print("Checkpoint 10: Displaying first 5 rows of prediction_df")

prediction_df = pd.DataFrame(results, columns = ['image_id', 'PredictionString', 'score', 'box_xmin', 'box_ymin', 'box_xmax', 'box_ymax'])
print(prediction_df.head(5)) # Observe output data frame

test_df['PredictionString'] = prediction_df['PredictionString']
test_df['score'] = prediction_df['score']
test_df['box_xmin'] = prediction_df['box_xmin']
test_df['box_ymin'] = prediction_df['box_ymin']
test_df['box_xmax'] = prediction_df['box_xmax']
test_df['box_ymax'] = prediction_df['box_ymax']

if (__name__ == '__main__') & (show_test_image_flag == True):
    print("Showing image: ", image_ids[wanted])
    sample = images[wanted].permute(1,2,0).cpu().numpy()
    boxes = outputs[wanted]['boxes'].data.cpu().numpy()
    scores = outputs[wanted]['scores'].data.cpu().numpy()

    box = boxes[0].astype(np.int32)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    sample = np.ascontiguousarray(sample)
    
    cv2.rectangle(sample,
                (box[0], box[1]),
                (box[2], box[3]),
                (220, 0, 0), 2)
        
    ax.set_axis_off()
    ax.imshow(sample)
    plt.show(block = True)

print("Checkpoint 11: Saving Predictions to test_df.csv [Final]")

test_df.to_csv("test_df.csv", index = False)
