from easydict import EasyDict as edict
import json

# config = edict()
# config.TRAIN = edict()

## Adam
batch_size = 16
lr_init1 = 1e-4 #学习率
lr_init2 = 1e-5 #学习率
beta1 = 0.9 #momentum

## initialize G
n_epoch_init = 6 #??????????
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)
alpha = 0.001
beta = 0.01

## adversarial learning (SRGAN)
n_epoch = 6
lr_decay = 0.1 #？？？？
decay_every = int(n_epoch / 2)

## train set location
hr_img_path = 'VOCROOT/VOC2007/JPEGImages'
crop_img_path = 'VOCROOT/VOC2007/Crop_img'
crop_img_96 = 'VOCROOT/VOC2007/Crop_img_96'
hr_labcor_path = 'VOCROOT/VOC2007/labels'
hr_lab2_path = 'VOCROOT/VOC2007/labels+.txt'
hr_lab_path = 'VOCROOT/VOC2007/labels.txt'

# config.VALID = edict()
## test set location
# config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
lr_img_path = 'VOCROOT/VOC2007/val_img'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")



num_parallel_calls = 4
input_shape = 416
max_boxes = 20
jitter = 0.3
norm_decay = 0.99
norm_epsilon = 1e-3
pre_train = True
num_anchors = 9
num_classes = 20
training = True
ignore_thresh = .5
learning_rate = 0.001
train_batch_size = 10
val_batch_size = 10
train_num = 118287
val_num = 5000
Epoch = 50
obj_threshold = 0.3
nms_threshold = 0.5
gpu_index = "0"
log_dir = './logs_yolo'
data_dir = "dataset"
model_dir = './test_model/model.ckpt-192192'
pre_train_yolo3 = False
yolo3_weights_path = './dataset/yolov3.weights'
darknet53_weights_path = './dataset/darknet53.weights'
anchors_path = './dataset/yolo_anchors.txt'
classes_path = './dataset/voc_classes.txt'

# train_data_file = '/data0/dataset/coco/train2017'
# val_data_file = '/data0/dataset/coco/val2017'
# train_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_train2017.json'
# val_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_val2017.json'
