from torch.utils.data import Dataset

import glob
import os
import numpy as np
import torch

from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys



import cv2
import copy
import json
from glob import glob
from torch.utils.data.dataloader import default_collate

import pickle
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

class COCODetection(Dataset):
    def __init__(self, img_size=416, year = "2017", mode = "train", is_training = True):
        root_path = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO"
        self.image_path = os.path.join(root_path, "images", "{}{}".format(mode, year))
        self.img_files = glob("D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO\\images\\{}{}\\*.jpg".format(mode, year))
        self.ann_file = "D:\\WorkSpace\\JupyterWorkSpace\\DataSet\\Object-Detection\\COCO\\annotations\\instances_{}{}.json".format(mode, year)

        # ---------
        #  Parse Json to Label
        # COCO Bounding box: (x-top left, y-top left, width, height)
        # ---------

        dataset = json.load(open(self.ann_file, 'r'))
        # Parse Json
        image_data = {}
        invalid_anno = 0

        for image in dataset["images"]:
            if image["id"] not in image_data.keys():
                image_data[image["id"]] = {"file_name": image["file_name"], "objects": []}

        for ann in dataset["annotations"]:
            if ann["image_id"] not in image_data.keys():
                invalid_anno += 1
                continue
            # COCO Bounding box: (x-min, y-min, x-max, y-max)
            image_data[ann["image_id"]]["objects"].append(
                [int(ann["bbox"][0]), int(ann["bbox"][1]), int(ann["bbox"][0] + ann["bbox"][2]),
                int(ann["bbox"][1] + ann["bbox"][3]), ann["category_id"]])

            # COCO Bounding box: (x-top left, y-top left, width, height)
            # image_data[ann["image_id"]]["objects"].append(
            #     [int(ann["bbox"][0]), int(ann["bbox"][1]), int(ann["bbox"][2]), int(ann["bbox"][3]), ann["category_id"]])

        self.label_files = image_data  

        self.class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                          31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                          55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                          82, 84, 85, 86, 87, 88, 89, 90]  

        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"]                             
        # self.num_classes = len(self.classes)
        self.img_size = img_size
        self.batch_count = 0
        self.is_training = is_training
        print('label_files', len(self.label_files))
        print('img_files',len(self.img_files))

    def __len__(self):
        return len(self.img_files)    

    def __getitem__(self, index):
        # if index in self.label_files:
        _id = list(self.label_files.keys())[index]
        img_fn = self.label_files[_id]["file_name"]        
        # ---------
        #  Image
        # ---------
        img_path = os.path.join(self.image_path, img_fn)
        img = cv2.imread(img_path)
        # ---------
        #  Label
        # COCO Bounding box: (x-top left, y-top left, width, height)
        # ---------
        image_dict = self.label_files[_id]       
        objects = copy.deepcopy(image_dict["objects"])
        for idx in range(len(objects)):
            objects[idx][4] = self.class_ids.index(objects[idx][4])   
            objects[idx][2] = objects[idx][2] - objects[idx][0]
            objects[idx][3] = objects[idx][3] - objects[idx][1]
             
        return img, objects
 
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"] 
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




class COCOModule(pl.LightningDataModule):
    def __init__(self, batch_size, img_size=416):
        super().__init__()
        self.batch_size = batch_size
        self.img_size = img_size
        self.name = "COCO"
        self.year = "2014"
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
           "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
           "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
           "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
           "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
           "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
           "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
           "teddy bear", "hair drier", "toothbrush"] 

    def setup(self, stage):      
        if stage == 'fit' or stage is None:
            # mode = "train" #val test
            # year = "2017" #2014
            self.train_dataset = COCODetection(self.img_size, year = self.year, mode = "train", is_training = True)
            self.val_dataset = COCODetection(self.img_size, year = self.year, mode = "val", is_training = False)

            self.train_dataset = DatasetFromSubset(
                self.train_dataset, 
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
                self.val_dataset, 
                transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
                ], bbox_params=A.BboxParams(format='coco'))
            )

            self.num_classes = len(self.train_dataset.classes)
            # self.num_classes = self.train_dataset.num_classes
            # self.val_dataset.is_training = False
        if stage == 'test' or stage is None:
            # self.test_dataset = COCODetection(self.img_size, year = self.year, mode = "val", is_training = False)
            self.test_dataset = COCODetection(self.img_size, year = self.year, mode = "val", is_training = False)
            self.test_dataset = DatasetFromSubset(
                self.test_dataset, 
                transform = A.Compose([
                    A.Resize(self.img_size, self.img_size),
                    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
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
                num_workers=0,# 加载数据的进程个数
                collate_fn=self.train_dataset.collate_fn,
                drop_last=True
            )

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return DataLoader(
                dataset=self.val_dataset,# TensorDataset类型数据集
                batch_size=self.batch_size,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=0,# 加载数据的进程个数
                collate_fn=self.val_dataset.collate_fn,
                drop_last=True
            )

    def test_dataloader(self):
        return DataLoader(
                dataset=self.test_dataset,# TensorDataset类型数据集
                batch_size=1,# mini batch size
                shuffle=False,# 设置随机洗牌
                num_workers=0,# 加载数据的进程个数
                collate_fn=self.test_dataset.collate_fn,
                drop_last=True
            )
    def get_class(self):
        return self.classes

