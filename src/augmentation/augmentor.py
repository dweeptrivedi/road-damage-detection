#%matplotlib inline
#import matplotlib.pyplot as plt
import cv2

import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import argparse
from pascal_voc_writer import Writer


def augmentor(input_file="train.txt", num_augmentors=3):
    t1 = time.time()
    ia.seed(1)

    # read inputfile
    bbox_dict = defaultdict(list)
    imgFiles = []
    with open(input_file,"r") as f:
        imgFiles = f.readlines()
        for i in range(len(imgFiles)):
            imgFiles[i] = imgFiles[i].strip()
            assert(os.path.isfile(imgFiles[i]))
    print("total files:", len(imgFiles))

    # read bounding boxes for each image
    for img_file in imgFiles:
        xml_file = img_file.replace("JPEGImages","Annotations").rsplit(".",1)[0]+".xml"
        tree = ET.parse(xml_file)
        root = tree.getroot()

        #loop over each object tag in annotation tag
        for objects in root.findall('object'):
            surfaceType = objects.find('name').text
            bndbox = objects.find('bndbox')
            [minX,minY,maxX,maxY] = [int(child.text) for child in bndbox]
            bbox_dict[img_file].append([surfaceType, minX, minY, maxX, maxY])

    #read image and attach bounding boxes associated with that image
    #this is required since we want to transform bounding boxes along with the image
    j = 0
    images_path = []
    bbs = []
    images = []
    for img in bbox_dict:
        if j<len(bbox_dict):
            ia_bbs = []
            for bbox in bbox_dict[img]:
                ia_bbs.append(ia.BoundingBox(x1=bbox[1], y1=bbox[2], x2=bbox[3], y2=bbox[4],label=bbox[0]))
            image = cv2.imread(img).astype(np.uint8)
            ia_bbs_for_image = ia.BoundingBoxesOnImage(ia_bbs, shape=image.shape)
            images_path.append(img)
            images.append(image)
            bbs.append(ia_bbs_for_image)
        j+=1

    t2 = time.time()
    print("time taken to construct image list:", t2-t1)


    t1 = time.time()
    #currently we are using 3 types of augmentors to augment each image
    aug=[0 for c in range(num_augmentors)]
    aug[0] = iaa.Grayscale(alpha=(0.5, 1.0))
    aug[1] = iaa.Sharpen(alpha=(0.0, 0.25), lightness=(0.75, 1.5))
    aug[2] = iaa.Multiply((0.75, 1.25))
    for i in range(3,num_augmentors):
        aug[i] = aug[i%3]

    for i in range(len(aug)):
        aug[i] = aug[i].to_deterministic()

    images_aug = []
    bbs_aug = []
    for i in range(len(aug)):
        print("augmentor:",i)
        images_aug = aug[i].augment_images(images)
        bbs_aug = aug[i].augment_bounding_boxes(bbs)
        for j in range(len(bbs_aug)):
            bbs_aug[j] = bbs_aug[j].remove_out_of_image().cut_out_of_image()

        for k in range(len(images_aug)):
            writer = Writer(images_path[k], images_aug[k].shape[1], images_aug[k].shape[0])
            for l in range(len(bbs_aug[k].bounding_boxes)):
                after = bbs_aug[k].bounding_boxes[l]
                writer.addObject(after.label, int(after.x1),int(after.y1),int(after.x2),int(after.y2))       
            xml_path = images_path[k].replace("JPEGImages","Augmented_dataset")
            xml_path = xml_path.rsplit('.',1)[0]
            if i==0 and (not os.path.isdir(xml_path)):
                os.makedirs(xml_path)
            xml_file_name = os.path.basename(images_path[k]).rsplit(".",1)[0]+"_"+str(i+3)+"_aug_noncrop.xml"
            xml_file = os.path.join(xml_path,xml_file_name)
            jpeg_file = xml_file.rsplit(".",1)[0]+"."+images_path[k].rsplit(".",1)[1]
            writer.save(xml_file)
            cv2.imwrite(jpeg_file,images_aug[k])

    t2 = time.time()
    print("generated augmented images in: ",t2-t1," seconds")
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='generate augmanted images.')
    parser.add_argument('--input', type=str, help='file containing path to images that needs to be augmented',default='train.txt')
    parser.add_argument('--num', type=int, help='number of augmented images per input image',default=3)
    args = parser.parse_args()

    augmentor(input_file=args.input, num_augmentors=args.num)
