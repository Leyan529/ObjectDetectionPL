import torch.nn as nn
import torch
import math
from torch.nn import functional as F

from LightningFunc.accuracy import xywh2xyxy, bbox_iou

import pytorch_lightning as pl

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.lightningUtils import *
from LightningFunc.losses import configure_loss
import pickle

class YOLOv2(pl.LightningModule):
    anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)]
    img_size = 416
    grid_size = 0
    ignore_thres = 0.5
    colors = pickle.load(open("dataset//pallete", "rb"))   
    anch_masks = None
    def __init__(self, classes, args):
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)        
        self.args = args 
        self.__build_model()
        self.__build_func(YOLOv2)   
        self.sample = (1, 3, self.img_size, self.img_size)
        self.sampleImg=torch.rand(self.sample).cuda()

        self.criterion = configure_loss(args, None, self.anchors, None, self.num_classes, self.img_size)

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname)

    def __build_model(self):
        
        # LAYER 1 (STAGE 1)
        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        
        # LAYER 2 (STAGE 1)
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        
        # LAYER 3 (STAGE 1)
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        # LAYER 4 (STAGE 1)
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        # LAYER 5 (STAGE 1)
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))        
        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        
        # LAYER 6 (STAGE 1)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        
        # LAYER 7 (STAGE 2)
        self.stage2_a_conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_b_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.1, inplace=True))        
        
        # LAYER 8 (STAGE 3)
        self.stage3_conv1 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * (5 + self.num_classes), 1, 1, 0, bias=False)

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "YOLOv2"
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
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.contiguous().view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)

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
                # answers = int(prediction.size(1) /3) # [x, y, w, h, conf, numclass...] = 5 + C
                answers = int(prediction.size(1) /5) # [x, y, w, h, conf, numclass...] = 5 + C

                prediction = (
                    # x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                    prediction.view(num_samples, 5, answers, grid_size, grid_size)
                    .permute(0, 1, 3, 4, 2)
                    .contiguous()
                )
                prediction = prediction.view(num_samples, -1 , answers) # torch.Size([1, 845, 25])
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

   