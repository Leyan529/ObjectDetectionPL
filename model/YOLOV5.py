import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from LightningFunc.utils.YoloV5Utils import *

from LightningFunc.accuracy import xywh2xyxy, bbox_iou_v5

import pytorch_lightning as pl

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.lightningUtils import *
from LightningFunc.losses import configure_loss
import pickle

class backbone_head(nn.Module):    
    def __init__(self, num_classes, num_anchors, yolov5_type):               
        super().__init__()
        if yolov5_type == "Yolov5s":
            depth_multiple = 0.33
            width_multiple = 0.5
        elif yolov5_type == "Yolov5m":
            depth_multiple = 0.67
            width_multiple = 0.75
        elif yolov5_type == "Yolov5l":
            depth_multiple = 1.0
            width_multiple = 1.0
        elif yolov5_type == "Yolov5x":
            depth_multiple = 1.33
            width_multiple = 1.25                        

        self.no = (num_classes + 5) * num_anchors
        self.seq0_Focus = Focus(3, 64, 3, width_multiple=width_multiple)
        self.seq1_Conv = Conv(64, 128, 3, 2, width_multiple=width_multiple)
        self.seq2_Bottleneck = Bottleneck(128, 128, width_multiple=width_multiple)
        self.seq3_Conv = Conv(128, 256, 3, 2, width_multiple=width_multiple)
        self.seq4_BottleneckCSP = BottleneckCSP(256, 256, 9, width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.seq5_Conv = Conv(256, 512, 3, 2, width_multiple=width_multiple)
        self.seq6_BottleneckCSP = BottleneckCSP(512, 512, 9, width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.seq7_Conv = Conv(512, 1024, 3, 2, width_multiple=width_multiple)
        self.seq8_SPP = SPP(1024, 1024, [5, 9, 13], width_multiple=width_multiple)
        self.seq9_BottleneckCSP = BottleneckCSP(1024, 1024, 6, width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.seq10_BottleneckCSP = BottleneckCSP(1024, 1024, 3, False, width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.seq11_Conv2d = nn.Conv2d(int(round(1024*width_multiple,1)), self.no, 1, 1, 0)
        self.seq14_Conv = Conv(1536, 512, 1, 1, width_multiple=width_multiple)
        self.seq15_BottleneckCSP = BottleneckCSP(512, 512, 3, False, width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.seq16_Conv2d = nn.Conv2d(int(round(512*width_multiple,1)), self.no, 1, 1, 0)
        self.seq19_Conv = Conv(768, 256, 1, 1, width_multiple=width_multiple)
        self.seq20_BottleneckCSP = BottleneckCSP(256, 256, 3, False, width_multiple=width_multiple, depth_multiple=depth_multiple)
        self.seq21_Conv2d = nn.Conv2d(int(round(256*width_multiple,1)), self.no, 1, 1, 0)

    def forward(self, x):
        x = self.seq0_Focus(x)
        x = self.seq1_Conv(x)
        x = self.seq2_Bottleneck(x)
        x = self.seq3_Conv(x)
        xRt0 = self.seq4_BottleneckCSP(x)
        x = self.seq5_Conv(xRt0)
        xRt1 = self.seq6_BottleneckCSP(x)
        x = self.seq7_Conv(xRt1)
        x = self.seq8_SPP(x)
        x = self.seq9_BottleneckCSP(x)
        route = self.seq10_BottleneckCSP(x)
        out0 = self.seq11_Conv2d(route)
        route2 = F.interpolate(route, size=(int(route.shape[2] * 2), int(route.shape[3] * 2)), mode='nearest')
        x = torch.cat([route2, xRt1], dim=1)
        x = self.seq14_Conv(x)
        route = self.seq15_BottleneckCSP(x)
        out1 = self.seq16_Conv2d(route)
        route2 = F.interpolate(route, size=(int(route.shape[2] * 2), int(route.shape[3] * 2)), mode='nearest')
        x = torch.cat([route2, xRt0], dim=1)
        x = self.seq19_Conv(x)
        x = self.seq20_BottleneckCSP(x)
        out2 = self.seq21_Conv2d(x)
        return out2, out1, out0

class Yolo_Layers(nn.Module):
    def __init__(self, nc=80, anchors=()):  # detection layer
        super(Yolo_Layers, self).__init__()
        self.stride = torch.tensor([ 8., 16., 32.]).cuda()  # strides computed during build
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid

        self.anchor_grid = torch.tensor(anchors).float().view(self.nl, 1, -1, 1, 1, 2).cuda()
        self.anchors = self.anchor_grid.view(self.nl, -1, 2) / self.stride.view(-1, 1, 1)

    def forward(self, x, inference):
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        
class YOLOv5(pl.LightningModule):
    # def __init__(self, num_classes, anchors=(), training=False):
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anch_masks = None
    img_size = 640
    # grid_size = 0
    ignore_thres = 0.5
    colors = pickle.load(open("dataset//pallete", "rb"))
    inference = False
    def __init__(self, classes, args):
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)

        self.args = args 
        self.__build_model()
        self.__build_func(YOLOv5)   
        self.sample = (2, 3, self.img_size, self.img_size)
        self.sampleImg=torch.rand(self.sample).cuda()

        self.criterion = configure_loss(args, None, self.anchors, None, self.num_classes, self.img_size)

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname)

    def __build_model(self):
        self.num_anchors = len(self.anchors[0]) // 2 
        self.backbone_head = backbone_head(self.num_classes, self.num_anchors, self.args.type)
        self.yolo_layers = Yolo_Layers(self.num_classes, anchors=self.anchors)

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "YOLOv5"
        setattr(obj, "training_step", training_step)
        setattr(obj, "training_epoch_end", training_epoch_end)
        setattr(obj, "validation_step", validation_step)
        setattr(obj, "validation_epoch_end", validation_epoch_end)
        setattr(obj, "test_step", test_step)
        setattr(obj, "test_epoch_end", test_epoch_end)
        setattr(obj, "configure_optimizers", configure_optimizers)
        setattr(obj, "mark_target", mark_target)   
        setattr(obj, "mark_pred", mark_pred)   
        setattr(obj, "saveDetail", saveDetail) 
        setattr(obj, "write_Best_model_path", write_Best_model_path)
        setattr(obj, "read_Best_model_path", read_Best_model_path) 
        setattr(obj, "get_yolo_statistics", get_yolo_statistics) 
        
    def forward(self, x):
        out2, out1, out0 = self.backbone_head(x)
        output = self.yolo_layers([out2, out1, out0], self.inference)
        return output

    def non_max_suppression(self, predictions, conf_thres=0.5, nms_thres=0.4):
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """
        conf_thres = -0.0151
        # if type(predictions) != list: 
        #     predictions = [predictions]
        #     # Anchor = 3
        #     # 507     3*13*13
        #     # 2028   3*26*26
        #     # 8112   3*52*52
        #     # [batch_size, 3*(13+6),52*52]
        predictions_list = []
        for prediction in predictions:
            num_samples = prediction.size(0)
            grid_size = prediction.size(2)
            answers = prediction.size(4)

            prediction = (
                prediction.view(num_samples, 3, answers, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            prediction = prediction.view(num_samples, -1 , answers)
            predictions_list.append(prediction)

        prediction = torch.cat(predictions_list, dim=1)
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
        output = [None for _ in range(len(prediction))]
        for image_i, image_pred in enumerate(prediction):
            # Filter out confidence scores below threshold
            # image_pred = image_pred[image_pred[:, 4] >= -0.0151]
            image_pred = image_pred[image_pred[:, 4] >= conf_thres]
            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Object confidence times class confidence
            score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
            # Sort by it
            image_pred = image_pred[(-score).argsort()]
            class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
            detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
            # Perform non-maximum suppression
            keep_boxes = []
            while detections.size(0):
                large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
                label_match = detections[0, -1] == detections[:, -1]
                # Indices of boxes with lower confidence scores, large IOUs and matching labels
                invalid = large_overlap & label_match
                weights = detections[invalid, 4:5]
                # Merge overlapping bboxes by order of confidence
                detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
                keep_boxes += [detections[0]]
                detections = detections[~invalid]
            if keep_boxes:
                output[image_i] = torch.stack(keep_boxes)

        return output

if __name__ == '__main__':    
    net = YOLOv5(80, None)
    
    