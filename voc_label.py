import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["1","2","8","10"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(wd, year, image_id):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(wd, year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    info = ""
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = "%s,%s,%s,%s,%d" % (xmlbox.find('xmin').text, xmlbox.find('ymin').text, xmlbox.find('xmax').text, xmlbox.find('ymax').text, cls_id)
        if len(info) == 0:
            info = b
        else:
            info = info + " " + b
    return info

# wd = getcwd()
wd = 'C:/Users/Administrator/Desktop/tensorflow-yolov3'

for year, image_set in sets:
    image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt'%(wd, year, image_set)).read().strip().split()
    list_file = open('./data/dataset/%s.txt'% image_set, 'w')
    count = 0
    for image_id in image_ids:
        filename = '%s/VOC%s/JPEGImages/%s.jpg '%(wd, year, image_id)
        info = convert_annotation(wd, year, image_id)
        list_file.write(filename + info + "\n")
        count+=1
    list_file.close()

