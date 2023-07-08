import os
import sys
#from git import Repo
import object_detection
import tensorflow as tf
#import tensorflow._api.v2.compat.v1 as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Création de la structure du dossier
for path in paths.values():
    if not os.path.exists(path):
        if os.name == 'posix':
            os.mkdir(path)
        if os.name == 'nt':
            os.mkdir(path)

# Download Tensorflow pretrained models and Install TFOD(Tensorflow object detection)

if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
    #Repo.clone_from('https://github.com/tensorflow/models', paths['APIMODEL_PATH'])
    os.system("git clone https://github.com/tensorflow/models " + paths['APIMODEL_PATH'])

""" # Install TFOD
if os.name=='posix':  
    !apt-get install protobuf-compiler
    !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .  """
    

# VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
# Verify Installation
# python {VERIFICATION_SCRIPT}
# os.system(VERIFICATION_SCRIPT)

if os.name =='posix':
    os.system("wget " + PRETRAINED_MODEL_URL)
    os.system("mv " + PRETRAINED_MODEL_NAME+".tar.gz "+ paths['PRETRAINED_MODEL_PATH'])
    os.system("cd " + paths['PRETRAINED_MODEL_PATH'] + " && tar -zxvf "+PRETRAINED_MODEL_NAME+'.tar.gz')

labels = [{'name':'♥ 1', 'id':1}, {'name':'♥ 2', 'id':2}, {'name':'♥ 3', 'id':3}, {'name':'♥ 4', 'id':4}, {'name':'♥ 5', 'id':5},
          {'name':'♥ 6', 'id':6}, {'name':'♥ 7', 'id':7}, {'name':'♥ 8', 'id':8}, {'name':'♥ 9', 'id':9}, {'name':'♥ 10', 'id':10},
          {'name':'♥ J', 'id':11}, {'name':'♥ Q', 'id':12}, {'name':'♥ K', 'id':13},

          {'name':'♠ 1', 'id':14}, {'name':'♠ 2', 'id':15}, {'name':'♠ 3', 'id':16}, {'name':'♠ 4', 'id':17}, {'name':'♠ 5', 'id':18},
          {'name':'♠ 6', 'id':19}, {'name':'♠ 7', 'id':20}, {'name':'♠ 8', 'id':21}, {'name':'♠ 9', 'id':22}, {'name':'♠ 10', 'id':23},
          {'name':'♠ J', 'id':24}, {'name':'♠ Q', 'id':25}, {'name':'♠ K', 'id':26},

          {'name':'♦ 1', 'id':27}, {'name':'♦ 2', 'id':28}, {'name':'♦ 3', 'id':29}, {'name':'♦ 4', 'id':30}, {'name':'♦ 5', 'id':31},
          {'name':'♦ 6', 'id':32}, {'name':'♦ 7', 'id':33}, {'name':'♦ 8', 'id':34}, {'name':'♦ 9', 'id':35}, {'name':'♦ 10', 'id':36},
          {'name':'♦ J', 'id':37}, {'name':'♦ Q', 'id':38}, {'name':'♦ K', 'id':39},

          {'name':'♣ 1', 'id':40}, {'name':'♣ 2', 'id':41}, {'name':'♣ 3', 'id':42}, {'name':'♣ 4', 'id':43}, {'name':'♣ 5', 'id':44},
          {'name':'♣ 6', 'id':45}, {'name':'♣ 7', 'id':46}, {'name':'♣ 8', 'id':47}, {'name':'♣ 9', 'id':48}, {'name':'♣ 10', 'id':49},
          {'name':'♣ J', 'id':50}, {'name':'♣ Q', 'id':51}, {'name':'♣ K', 'id':52},
          
          {'name':'🔝', 'id':53}, {'name':'🃏', 'id':54}]

with open(files['LABELMAP'], 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

""" 
!cp '/content/drive/MyDrive/TFOD images/imgs.tar.gz' {paths['IMAGE_PATH']}
   
uncompress the compressed file

ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'imgs.tar.gz')
if os.path.exists(ARCHIVE_FILES):
  !tar -zxvf {ARCHIVE_FILES}
  !mv '/content/test' '/content/train' {paths['IMAGE_PATH']} """

# Get TF record script
if not os.path.exists(files['TF_RECORD_SCRIPT']):
    #Repo.clone_from('https://github.com/nicknochnack/GenerateTFRecord', paths['SCRIPTS_PATH'])
    os.system("git clone https://github.com/nicknochnack/GenerateTFRecord " + paths['SCRIPTS_PATH'])

os.system("python3 " + files['TF_RECORD_SCRIPT']+" -x "+ os.path.join(paths['IMAGE_PATH'], 'train')+ " -l "+files['LABELMAP']+" -o "+os.path.join(paths['ANNOTATION_PATH'], 'train.record'))
os.system("python3 " + files['TF_RECORD_SCRIPT']+" -x "+ os.path.join(paths['IMAGE_PATH'], 'test')+ " -l "+files['LABELMAP']+" -o "+os.path.join(paths['ANNOTATION_PATH'], 'test.record'))

# Copy Model config to training folder
if os.name =='posix':
    os.system("cp " + os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config') + " " + os.path.join(paths['CHECKPOINT_PATH']))

# Update config for transfer Learning
config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
config

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 54
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)

# Train the model
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')
#command = "python3 --model_dir='"+paths['APIMODEL_PATH']+"' --pipeline_config_path='"+files['PIPELINE_CONFIG']+"' --num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
command = "python3 {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000".format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])
 
print(command)
os.system(command)
