import tensorflow as tf
import os
import wget
import git
import object_detection
import cv2
import numpy as np

from matplotlib import pyplot as plt
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

# Setting up paths for model 
CUSTOM_MODEL_NAME = 'tensorflow_rcnn_model'
PRETRAINED_MODEL_NAME = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'SCRIPTS_PATH': os.path.join('../tensorflow_model', 'scripts'),
    'APIMODEL_PATH': os.path.join('../tensorflow_model','api_models'),
    'ANNOTATION_PATH': os.path.join('../../../../12-ocr-image-data','annotations'),
    'IMAGE_PATH': os.path.join('../../../../12-ocr-image-data','images'),    
    'MODEL_PATH': os.path.join('../tensorflow_model', 'models'),
    'PRETRAINED_MODEL_PATH': os.path.join('../tensorflow_model','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('../tensorflow_model','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('../tensorflow_model','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('../tensorflow_model','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('../tensorflow_model','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('../tensorflow_model','protoc')
}

files = {
    'PIPELINE_CONFIG':os.path.join('../tensorflow_model','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

for path in paths.values():
    if not os.path.exists(path):
        os.makedirs(path)
        
        

# Clone Tensorflow object detection repository (run in terminal)
""" if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'object_detection')):
        git.Repo.clone_from('https://github.com/tensorflow/models/', paths['APIMODEL_PATH']) """


# Download pretrained model (run in terminal)
"""wget.download(PRETRAINED_MODEL_URL)
os.rename(f'{PRETRAINED_MODEL_NAME}.tar.gz', paths['PRETRAINED_MODEL_PATH'] + '/' + PRETRAINED_MODEL_NAME) """


# Install Tensorflow Object Detection (run in terminal)
""" url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
wget.download(url)
os.rename (f"protoc-3.15.6-win64.zip", paths['PROTOC_PATH'] + '/' + 'protoc-3.15.6-win64.zip')
!cd tensorflow_model/protoc && tar -xf protoc-3.15.6-win64.zip
os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
!cd tensorflow_model/api_models/research && protoc object_detection/protos/*.proto --python_out=. && pip install . && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install
!cd tensorflow_model/api_models/research/slim && pip install -e .  """

# labelling
labels = [{'name':'weight', 'id':1}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')
        
# generating TFrecords (run in terminal)
""" git.Repo.clone_from('https://github.com/nicknochnack/GenerateTFRecord', paths['SCRIPTS_PATH']) """

# creating tfrecord files for train & test data (run in terminal)
""" python scripts/GenerateTFRecord/generate_tfrecord.py -x ../../../../12-ocr-image-data/images/train -l ../../../../12-ocr-image-data/annotations/label_map.pbtxt -o ../../../../12-ocr-image-data/annotations/train.record
python scripts/GenerateTFRecord/generate_tfrecord.py -x ../../../../12-ocr-image-data/images/test -l ../../../../12-ocr-image-data/annotations/label_map.pbtxt -o ../../../../12-ocr-image-data/annotations/test.record
 """
 
# copy model config to training folder
""" cp ../tensorflow_model/pre-trained-models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/pipeline.config ../tensorflow_model/models/tensorflow_rcnn_model """ 


# update config for transfer learning
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
config 

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  
    
pipeline_config.model.faster_rcnn.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]
config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)
    
    
""" # Train the model 
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=10000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
print(command)
os.system(command) """

# Evaluate the model
""" TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])
print(command)
os.system(command) """

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model) 
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

# Detectiong from an image
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'test', '20231006_083526.jpg')
img = cv2.imread(IMAGE_PATH)
#image_np = dict(enumerate(np.array(img).flatten(), 1))
image_np = np.array(img)
#print(image_np)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
#image_np_with_detections = dict(enumerate(np.array(img).flatten(), 1)).copy()
#image_np_with_detections_str = ''.join(map(str, image_np_with_detections))
#image_np_with_detections_int = int(image_np_with_detections_str)
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            #image_np_with_detections_int,
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
#plt.imshow(cv2.imread("Show image @ line 171", cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)))
plt.show()
#cv2.imshow("Show image @ line 171", cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
#cv2.waitKey(0)

detections.keys()

print('heelo world')