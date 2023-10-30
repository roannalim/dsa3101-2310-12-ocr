import tensorflow
import os
import wget
import git
import object_detection

# Setting up paths for model 
CUSTOM_MODEL_NAME = 'tensorflow_ssd_model'
PRETRAINED_MODEL_NAME = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'SCRIPTS_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model', 'scripts'),
    'APIMODEL_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','api_models'),
    'ANNOTATION_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/12-ocr-image-data','annotations'),
    'IMAGE_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/12-ocr-image-data','images'),    
    'MODEL_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('Desktop/Y3S1/DSA3101/project/tensorflow_model','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)
        
    
for file in files.values():
    if not os.path.exists(file):
        os.mkdir(file)
        

# Clone Tensorflow object detection repository
""" if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'object_detection')):
    git.Repo.clone_from('https://github.com/tensorflow/models/', paths['APIMODEL_PATH']) """

# Download pretrained model
""" wget.download(PRETRAINED_MODEL_URL)
    os.rename(f'Downloads/{PRETRAINED_MODEL_NAME}', paths['PRETRAINED_MODEL_PATH'] + '/' + PRETRAINED_MODEL_NAME) """



