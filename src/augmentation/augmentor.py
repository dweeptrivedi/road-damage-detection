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


class Augmentor(object):
    """Augmentor class to augment images for object detection task
    """
    
    def __init__(self, input_file="train.txt"):
        """init method
        
        Read the input file containing list of images to augment and bounding boxes of each image.
        
        Args:
            input_file (str): file containing paths to images
        
        """
        ia.seed(1)
        self.input_file = input_file
                
        # read inputfile
        self.bbox_dict = defaultdict(list)
        self.imgFiles = []
        self.aug = []
        with open(input_file,"r") as f:
            self.imgFiles = f.readlines()
            for i in range(len(self.imgFiles)):
                self.imgFiles[i] = self.imgFiles[i].strip()
                assert(os.path.isfile(self.imgFiles[i]))
        print("total files:", len(self.imgFiles))
        
        # read bounding boxes for each image
        for img_file in self.imgFiles:
            xml_file = img_file.replace("JPEGImages","Annotations").rsplit(".",1)[0]+".xml"
            tree = ET.parse(xml_file)
            root = tree.getroot()

            #loop over each object tag in annotation tag
            for objects in root.findall('object'):
                surfaceType = objects.find('name').text
                bndbox = objects.find('bndbox')
                [minX,minY,maxX,maxY] = [int(child.text) for child in bndbox]
                self.bbox_dict[img_file].append([surfaceType, minX, minY, maxX, maxY])

    
    
    def readImages(self):
        """load images and attach bounding boxes associated witht that images in imgaug
        """
        #read image and attach bounding boxes associated with that image
        #this is required since we want to transform bounding boxes along with the image
        self.images_path = []
        self.bbs = []
        self.images = []
        for img in self.bbox_dict:
            ia_bbs = []
            for bbox in self.bbox_dict[img]:
                ia_bbs.append(ia.BoundingBox(x1=bbox[1], y1=bbox[2], x2=bbox[3], y2=bbox[4],label=bbox[0]))
            image = cv2.imread(img).astype(np.uint8)
            ia_bbs_for_image = ia.BoundingBoxesOnImage(ia_bbs, shape=image.shape)
            self.images_path.append(img)
            self.images.append(image)
            self.bbs.append(ia_bbs_for_image)

    def addAugmentor(self, augmentor):
        """adds augmentor to current list of augmentors
        
        User can add any number of "imgaug.augmenters" objects, these are the
        transformations that will be applied to images and bounding boxes
        
        Args:
            augmentor (:obj:augmenters): imgaug.augmenters object
        
        """
        self.aug.append(augmentor)
        
    def augment(self):
        """augment all images with augmentor and save augmented images and XML files
        """
        for i in range(len(self.aug)):
            self.aug[i] = self.aug[i].to_deterministic()

        images_aug = []
        bbs_aug = []
        for i in range(len(self.aug)):
            print("augmentor:",i)
            images_aug = self.aug[i].augment_images(self.images)
            bbs_aug = self.aug[i].augment_bounding_boxes(self.bbs)
            for j in range(len(bbs_aug)):
                bbs_aug[j] = bbs_aug[j].remove_out_of_image().cut_out_of_image()

            for k in range(len(images_aug)):
                writer = Writer(self.images_path[k], images_aug[k].shape[1], images_aug[k].shape[0])
                for l in range(len(bbs_aug[k].bounding_boxes)):
                    after = bbs_aug[k].bounding_boxes[l]
                    writer.addObject(after.label, int(after.x1),int(after.y1),int(after.x2),int(after.y2))       
                xml_path = self.images_path[k].replace("JPEGImages","Augmented_dataset")
                xml_path = xml_path.rsplit('.',1)[0]
                if i==0 and (not os.path.isdir(xml_path)):
                    os.makedirs(xml_path)
                xml_file_name = os.path.basename(self.images_path[k]).rsplit(".",1)[0]+"_"+str(i+3)+"_aug.xml"
                xml_file = os.path.join(xml_path,xml_file_name)
                jpeg_file = xml_file.rsplit(".",1)[0]+"."+self.images_path[k].rsplit(".",1)[1]
                writer.save(xml_file)
                cv2.imwrite(jpeg_file,images_aug[k])

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='generate augmanted images.')
    parser.add_argument('--input', type=str, help='file containing path to images that needs to be augmented. sample at "../../examples/train_rddc_sample.txt"',default='../../examples/train_rddc_sample.txt')
    parser.add_argument('--num', type=int, help='number of augmented images per input image',default=3)
    args = parser.parse_args()

    A = Augmentor(input_file=args.input)
    A.readImages()
    A.addAugmentor(iaa.Grayscale(alpha=(0.5, 1.0)))
    A.addAugmentor(iaa.Sharpen(alpha=(0.0, 0.25), lightness=(0.75, 1.5)))
    A.addAugmentor(iaa.Multiply((0.75, 1.25)))
    A.augment()
    
