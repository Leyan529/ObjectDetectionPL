import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from LightningFunc.utils.YoloV4Utils import *
from LightningFunc.accuracy import xywh2xyxy, bbox_iou

import pytorch_lightning as pl

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.lightningUtils import *
from LightningFunc.losses import configure_loss
import pickle

class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0
        self.scale_x_y = 1

        self.model_out = model_out

    def forward(self, output, target=None):
        if self.training:
            return output
        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward_dynamic(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask),scale_x_y=self.scale_x_y)

class Yolov4Head(nn.Module):
    def __init__(self, output_ch, num_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=num_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=num_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=num_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            return [x2, x10, x18]

class YOLOv4(pl.LightningModule):
    anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
    anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    img_size = 416
    grid_size = 0
    ignore_thres = 0.5
    colors = pickle.load(open("dataset//pallete", "rb"))
    def __init__(self, classes, args):
        super().__init__()       
        self.classes = classes
        self.num_classes = len(self.classes)
        self.args = args 
        self.__build_model()
        self.__build_func(YOLOv4)   
        self.sample = (1, 3, self.img_size, self.img_size)
        self.sampleImg=torch.rand(self.sample).cuda()

        self.criterion = configure_loss(args, None, self.anchors, self.anch_masks, self.num_classes, self.img_size)

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname)

    def __build_model(self):
        yolov4conv137weight=None
        inference=False
        output_ch = (4 + 1 + self.num_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neek = Neck(inference)
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neek)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        self.head = Yolov4Head(output_ch, self.num_classes, inference)

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "YOLOv4"
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

        
        

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neek(d5, d4, d3)

        output = self.head(x20, x13, x6)

        return output


    def non_max_suppression(self, predictions, conf_thres=0.5, nms_thres=0.4):
        """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x1, y1, x2, y2, object_conf, class_score, class_pred)
        """
        conf_thres = -0.0151
        if type(predictions) != list: 
            predictions = [predictions]
            # Anchor = 3
            # 507     3*13*13
            # 2028   3*26*26
            # 8112   3*52*52
            # [batch_size, 3*(13+6),52*52]
        predictions_list = []
        for prediction in predictions:
            num_samples = prediction.size(0)
            grid_size = prediction.size(2)
            answers = int(prediction.size(1) /3) # [x, y, w, h, conf, numclass...] = 5 + C

            prediction = (
                # x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
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


if __name__ == "__main__":  
    num_classes = 20
    model = YOLOv4(yolov4conv137weight=None, num_classes=num_classes, inference=True)
    print(model)