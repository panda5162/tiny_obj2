import ssd
import gan

from utils import *
import os
import tensorlayer as tl


path = './demo/000009.jpg'
ssd.ssd(path)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='train, evaluate')
args = parser.parse_args()
tl.global_flag['mode'] = args.mode

if tl.global_flag['mode'] == 'train':
    # img_path = './VOCROOT/VOC2007/JPEGImages/'
    # img_name = os.listdir(img_path)
    # for name in img_name:
    #     path = os.path.join(img_path, name)
    #     label_list, img_list = ssd.ssd(path)

    gan.train()

elif tl.global_flag['mode'] == 'evaluate':
    img_path = './VOCROOT/VOC2007TEST/JPEGImages/'
    img_name = os.listdir(img_path)
    for name in img_name:
        path = os.path.join(img_path, name)
        label_list, img_list = simple_ssd_demo.ssd_res(path)

    gan.evaluate(img_list, label_list)

else:
    raise Exception("Unknow --mode")


# if len(label_list) != len(img_list):
#     print("len(label_list) != len(img_list)")
#     exit()
# for i in range(len(label_list)):
#     gan.train(img_list[i], label_list[i])

