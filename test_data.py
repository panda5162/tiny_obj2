import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

import tensorlayer as tl

import os

from PIL import Image


from scipy.misc import imread, imsave, imshow, imresize

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

sets = [('2007', 'train'), ('2007', 'trainval'), ('2007', 'val')]
# classes = ['person', 'bicycle']

#----------------------- label+cor ----------------------


def labcor(labcor_path):
    labcor = tl.files.read_file(labcor_path)
    data_string = labcor.split()
    cl = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i, da in enumerate(data_string):
        if i % 5 == 0:
            cl.append(int(da))
            continue
        if (i - 1) % 5 == 0:
            x1.append(float(da))
            continue
        if (i - 2) % 5 == 0:
            y1.append(float(da))
            continue
        if (i - 3) % 5 == 0:
            x2.append(float(da))
            continue
        if (i - 4) % 5 == 0:
            y2.append(float(da))
            continue
    return cl, x1, y1, x2, y2


def convert(size, box):
    x1 = box[0]
    y1 = box[2]
    x2 = box[1]
    y2 = box[3]
    return (x1, y1, x2, y2)


def convert_annotation(year, image_id):
    in_file = open('VOCROOT/VOC%s/Annotations/%s.xml' % (year, image_id))  # （如果使用的不是VOC而是自设置数据集名字，则这里需要修改）
    out_file = open('VOCROOT/VOC%s/labels/%s.txt' % (year, image_id), 'w')  # （同上）
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCROOT/VOC%s/labels/' % (year)):
        os.makedirs('VOCROOT/VOC%s/labels/' % (year))
    image_ids = open('VOCROOT/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCROOT/VOC%s/JPEGImages/%s.jpg\n' % (wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

#----------------------- crop_hr_img----------------------
train_hr_img_list = sorted(tl.files.load_file_list(path='./VOCROOT/VOC2007/JPEGImages', regx='.*.jpg', printable=False))
train_hr_labcor_list = sorted(tl.files.load_file_list(path='VOCROOT/VOC2007/labels', regx='.*.txt', printable=False))
num = 0
for i, labcorr in enumerate(train_hr_labcor_list):
    lab, x1, y1, x2, y2 = labcor(os.path.join('VOCROOT/VOC2007/labels', labcorr))
    tmp_path = train_hr_img_list[i]
    tmp_img = Image.open(os.path.join('./VOCROOT/VOC2007/JPEGImages', tmp_path))
    # tmp_img2 = imread(os.path.join('./VOCROOT/VOC2007/JPEGImages', tmp_path))
    # f = open('./VOCROOT/VOC2007/labels.txt', 'a')
    # lab_str = str(lab)
    # f.write(lab_str + ' ')
    # f.close()

    for j in range(len(lab)):
        train_hr_img = tmp_img.crop((x1[j], y1[j], x2[j], y2[j]))
        t = str(num)
        num_str = t.zfill(6)
        train_hr_img.save('./VOCROOT/VOC2007/Crop_img/{}.jpg'.format(num_str))
        num = num + 1

#----------------------- get labels ----------------------
filenames = os.listdir('VOCROOT/VOC2007/labels')
f2 = open('VOCROOT/VOC2007/labels+.txt', 'w')

for filename in sorted(filenames):
    filepath = 'VOCROOT/VOC2007/labels'+'/'+filename
    for line in open(filepath):
        f2.writelines(line)
f2.close()

labb, _, _, _, _ = labcor('VOCROOT/VOC2007/labels+.txt')
f = open('VOCROOT/VOC2007/labels.txt', 'w')
for i, la in enumerate(labb):
    f.write(str(la) + ' ')
f.close()

#----------------------- crop img 384 ----------------------
img_path = 'VOCROOT/VOC2007/Crop_img/'
crop_img_96 = 'VOCROOT/VOC2007/Crop_img_96/'
count = os.listdir(img_path)
for i in range(0, len(count)):
    im = Image.open(img_path + str(i).zfill(6)+'.jpg')
    im = im.resize((96, 96))
    im.save(crop_img_96 + str(i).zfill(6) + '.jpg')

