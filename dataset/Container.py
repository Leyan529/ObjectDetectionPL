from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import random
import os
import sys
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F
import sys
import time
import datetime


import cv2
import copy
import json
from glob import glob
from torch.utils.data.dataloader import default_collate

import pickle
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pytorch_lightning as pl
from copy import deepcopy

import albumentations as A

colors = pickle.load(open("dataset//pallete", "rb"))
BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White

def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h, _ = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35, 
        color=(255, 255, 255), 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    img *= (0.229, 0.224, 0.225)
    img += (0.485, 0.456, 0.406)
    # img *= 255.0 
    # img = np.concatenate((img[:,:,-1:], img[:,:,1:2], img[:,:,0:1]), axis=2)
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
        # img = visualize_bbox(img, bbox, class_name, colors[int(category_id)])
    cv2.imshow("CoCo2", img)
    cv2.waitKey()
# ----------------------------------------------
class ContainerDetection(Dataset):
    def __init__(self, img_size=416, is_training = True):
        self.img_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\MosquitoContainer\\train_cdc\\train_images\\*.jpg")
        self.label_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\MosquitoContainer\\train_cdc\\train_annotations\\*.xml")

        self.classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet',
                        'tub', 'washing_machine', 'water_tower'] 
        self.num_classes = len(self.classes)
        self.img_size = img_size
        self.batch_count = 0
        self.is_training = True
        print('label_files',len(self.label_files))
        print('img_files',len(self.img_files))

    def __len__(self):
        return len(self.img_files)  

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        image_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(image_path)

        # # ---------
        # #  Label
        # # ---------
        image_xml_path = self.label_files[index % len(self.img_files)]
        if os.path.exists(image_xml_path):
            annot = ET.parse(image_xml_path)
            objects = []
            for obj in annot.findall('object'):
                xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                        ["xmin", "xmax", "ymin", "ymax"]]
                label = self.classes.index(obj.find('name').text.lower().strip())

                if xmin >= 0 and ymin >= 0 and (xmax-xmin)>=0 and (ymax-ymin)>=0 :
                    objects.append([xmin, ymin, xmax-xmin, ymax-ymin, label]) 
 
            return img, objects



class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet',
                        'tub', 'washing_machine', 'water_tower'] 
        self.view_mark = False

    def __getitem__(self, index):
        x, y = self.subset[index]

        transformed = self.transform(image=x, bboxes=y)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        category_ids = [list(x)[-1] for x in transformed_bboxes]
        # visualize(transformed_image, transformed_bboxes, category_ids, self.classes)

        tran_x, tran_y = transformed_image, transformed_bboxes
        output_image = deepcopy(tran_x)
        objects = deepcopy(tran_y)
        padded_w, padded_h, _ = output_image.shape

        for idx in range(len(objects)):
            boxes = objects[idx][:-1]
            # Calculate train: [x1, y1, w, h] from transform, then normalized scale(padded_h, padded_w)
            x1 =  boxes[0] / padded_w
            y1 =  boxes[1] / padded_h
            x2 =  boxes[2] / padded_w
            y2 =  boxes[3] / padded_h            

            boxes = [x1, y1, x2, y2]
            objects[idx] = [0, objects[idx][-1]] + boxes
        
        '''Test Mark'''
        if self.view_mark:
            for idx in range(len(objects)):
                boxes = objects[idx][1:]
                # orign_image = image
                xmin, ymin, xmax, ymax = int(boxes[1]*padded_w), int(boxes[2]*padded_h), int(boxes[3]*padded_w), int(boxes[4]*padded_h)
                xmax += xmin
                ymax += ymin
                cls_id = int(boxes[0])
                color = colors[cls_id]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2) # 邊框
                text_size = cv2.getTextSize(self.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                # cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, self.classes[cls_id],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                # print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax))    
            cv2.imshow("CoCo2", output_image)
            cv2.waitKey()
        return torch.Tensor(tran_x), torch.Tensor(np.array(objects, dtype=np.float32))

    def collate_fn(self, batch):
        items = list(zip(*batch))
        items[0] = default_collate(items[0])    
        items[0] = items[0]/255     # Normalize     
        for i, data in enumerate(items[1]):
            if data.shape[0] == 0: 
                continue
            data[:,0] = i
        items[1] = torch.cat(items[1], dim = 0)
        items[0] = items[0].permute(0, 3, 1, 2)
        return items[0], items[1]

    def __len__(self):
        return len(self.subset)


class MosquitoModule(pl.LightningDataModule):
    def __init__(self, bsz, img_size=416):
        super().__init__()
        self.batch_size = bsz
        self.img_size = img_size  
        self.name = "Mosquito"     
        self.classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet',
                        'tub', 'washing_machine', 'water_tower'] 

    def setup(self, stage):      
        if stage == 'fit' or stage is None:
            full_dataset = ContainerDetection(self.img_size)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size

            train_set, val_set = torch.utils.data.dataset.random_split(full_dataset, [train_size, test_size])

            self.train_dataset = DatasetFromSubset(
                train_set, 
                transform = A.Compose([
                        A.Resize(self.img_size, self.img_size),
                        A.HorizontalFlip(p=0.2),
                        A.VerticalFlip(p=0.2),
                        A.ShiftScaleRotate(p=0.2),
                        A.RandomBrightnessContrast(p=0.2),
                        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.2),
                        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255)
                    ], bbox_params=A.BboxParams(format='coco'))
            )
            self.val_dataset = DatasetFromSubset(
                val_set, 
                transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255)
                ], bbox_params=A.BboxParams(format='coco'))
            )

            self.num_classes = full_dataset.num_classes
            # self.val_dataset.is_training = False
        if stage == 'test' or stage is None:
            self.test_dataset = ContainerDetection(self.img_size)
            self.test_dataset = DatasetFromSubset(
                self.test_dataset, 
                transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255)
                ], bbox_params=A.BboxParams(format='coco'))
            )

    def prepare_data(self):
        # 在该函数里一般实现数据集的下载等，只有cuda:0 会执行该函数
        pass
    def train_dataloader(self):
        return DataLoader(
                dataset=self.train_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=True,# 设置随机洗牌
                num_workers=5,# 加载数据的进程个数
                collate_fn=self.train_dataset.collate_fn,
                drop_last=True
            )

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return DataLoader(
                dataset=self.val_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5,# 加载数据的进程个数
                collate_fn=self.val_dataset.collate_fn,
                drop_last=True
            )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,# TensorDataset类型数据集
                batch_size=1,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=5,# 加载数据的进程个数
                collate_fn=self.test_dataset.collate_fn,
                drop_last=True
            )
    def get_class(self):
        return self.classes

    