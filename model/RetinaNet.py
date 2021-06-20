import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_lightning as pl

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.lightningUtils import *
from LightningFunc.losses import configure_loss
import model.backbone.RetinaNetbone as bones
import pickle
from LightningFunc.utils.RetinaUtils import get_anchor_boxes


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class RetinaNet(pl.LightningModule):
    num_anchors = 9
    img_size = 600
    colors = pickle.load(open("dataset//pallete", "rb")) 
    def __init__(self, classes, args):
        super(RetinaNet, self).__init__()       
        self.classes = classes
        self.num_classes = len(self.classes)
        self.args = args 
        self.__build_model()
        self.__build_func(RetinaNet)   
        self.sample = (1, 3, self.img_size, self.img_size)
        self.sampleImg=torch.rand(self.sample).cuda()
        input_size = torch.Tensor([self.img_size, self.img_size])
        self.iou_boxes = get_anchor_boxes(input_size).cuda()    
        self.criterion = configure_loss(args, self.iou_boxes, None, None, self.num_classes, self.img_size)

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname)
        
    def __build_model(self):
        fpn=50
        if fpn == 50:
            self.fpn = bones.FPN(Bottleneck, [3,4,6,3])
        elif fpn == 101:
            self.fpn = bones.FPN(Bottleneck, [2,4,23,3])
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "RetinaNet"
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

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds,1), torch.cat(cls_preds,1), 

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def non_max_suppression(self, predictions, topk = 100, nms_thresh = 0.5, class_thresh = 0.45, mode='union'):
        loc_preds, cls_preds = predictions
        nms_boxes = []
        for bid in range(loc_preds.size(0)):
            loc_xy = loc_preds[bid, :,:2]
            loc_wh = loc_preds[bid, :,2:]

            xy = loc_xy * self.iou_boxes[:,2:] + self.iou_boxes[:,:2]
            wh = loc_wh.exp() * self.iou_boxes[:,2:]
            boxes = torch.cat([xy-wh/2, xy+wh/2], 1)  # [#anchors,4]

            score, labels = cls_preds[bid].sigmoid().max(1)          # [#anchors,]
            ids = score > class_thresh
            ids = ids.nonzero().squeeze()             # [#obj,]
            # keep = box_nms(boxes[ids], score[ids], threshold=NMS_THRESH)
            bboxes = boxes[ids]
            scores = score[ids]
            x1 = bboxes[:,0]
            y1 = bboxes[:,1]
            x2 = bboxes[:,2]
            y2 = bboxes[:,3]

            areas = (x2-x1+1) * (y2-y1+1)
            _, order = scores.sort(0, descending=True)
            order = order[:topk]

            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    break
                i = order[0].item()
                keep.append(i)           

                xx1 = x1[order[1:]].clamp(min=x1[i])
                yy1 = y1[order[1:]].clamp(min=y1[i])
                xx2 = x2[order[1:]].clamp(max=x2[i])
                yy2 = y2[order[1:]].clamp(max=y2[i])

                w = (xx2-xx1+1).clamp(min=0)
                h = (yy2-yy1+1).clamp(min=0)
                inter = w*h

                if mode == 'union':
                    ovr = inter / (areas[i] + areas[order[1:]] - inter)
                elif mode == 'min':
                    ovr = inter / areas[order[1:]].clamp(max=areas[i])
                else:
                    raise TypeError('Unknown nms mode: %s.' % mode)

                ids = (ovr<=nms_thresh).nonzero().squeeze()
                if ids.numel() == 0:
                    break
                order = order[ids+1]
            keep = torch.LongTensor(keep)
            nms_boxes.append(torch.cat([
                boxes[keep], 
                torch.zeros((len(keep),1)).type(torch.float32).cuda(),
                scores[keep].unsqueeze(1), 
                labels[keep].unsqueeze(1).type(torch.float32)
                ], dim = 1).cuda())
        # return torch.cat(nms_boxes, dim=0)
        return nms_boxes

