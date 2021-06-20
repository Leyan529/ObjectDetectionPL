import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg16
import pytorch_lightning as pl
from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.lightningUtils import *
from LightningFunc.losses import configure_loss
from LightningFunc.utils.SSDUtils import get_dboxes
import pickle


class SSD(pl.LightningModule):
    img_size = 300
    colors = pickle.load(open("dataset//pallete", "rb")) 
    def __init__(self, classes, args):
        super(SSD, self).__init__()

        self.classes = classes
        self.num_classes = len(self.classes)
        self.args = args 
        self.__build_model()
        self.__build_func(SSD)             
        self.sample = (1, 3, self.img_size, self.img_size)
        self.sampleImg=torch.rand(self.sample).cuda()
        # input_size = torch.Tensor([self.img_size, self.img_size])
        self.iou_boxes = get_dboxes().cuda()    
        # replace to original loss
        self.criterion = configure_loss(args, self.iou_boxes, None, None, self.num_classes, self.img_size)

        self.checkname = self.backbone
        self.dir = os.path.join("log_dir", self.args.data_module ,self.checkname)
        

        

    def __build_model(self):
        init_weights=True
        self.layers = []
        self.vgg_layers = []
        self.size = (300, 300)

        new_layers = list(vgg16(pretrained=True).features)

        # 將 VGG16 的 pool5 層從 size=2x2, stride=2 更改為 size=3x3, stride=1
        new_layers[16] = nn.MaxPool2d(2, ceil_mode=True)
        new_layers[-1] = nn.MaxPool2d(3, 1, padding=1)
        self.f1 = nn.Sequential(*new_layers[:23])
        self.vgg_layers.append(self.f1)


        self.cl1 = nn.Sequential(
            nn.Conv2d(512, 4*self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl1)

        self.bbx1 = nn.Sequential(
            nn.Conv2d(512, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx1)

        self.base1 = nn.Sequential(*new_layers[23:])
        self.vgg_layers.append(self.base1)

        # The refrence code uses a dilation of 6 which requires a padding of 6
        self.f2 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, dilation=3, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f2)

        self.cl2 = nn.Sequential(
            nn.Conv2d(1024, 6 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl2)

        self.bbx2 = nn.Sequential(
            nn.Conv2d(1024, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx2)

        self.f3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f3)

        self.cl3 = nn.Sequential(
            nn.Conv2d(512, 6 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl3)

        self.bbx3 = nn.Sequential(
            nn.Conv2d(512, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx3)

        self.f4 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f4)

        self.cl4 = nn.Sequential(
            nn.Conv2d(256, 6 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl4)

        self.bbx4 = nn.Sequential(
            nn.Conv2d(256, 6 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx4)

        self.f5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f5)

        self.cl5 = nn.Sequential(
            nn.Conv2d(256, 4 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl5)

        self.bbx5 = nn.Sequential(
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx5)

        self.f6 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=True)
        )
        self.layers.append(self.f6)

        self.cl6 = nn.Sequential(
            nn.Conv2d(256, 4 * self.num_classes, 3, padding=1)
        )
        self.layers.append(self.cl6)

        self.bbx6 = nn.Sequential(
            nn.Conv2d(256, 4 * 4, 3, padding=1)
        )
        self.layers.append(self.bbx6)

        if init_weights:
            self._init_weights(vgg_16_init=(not init_weights))
        
    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "SSD"
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
        out_cl = []
        out_bbx = []
        
        x1 = self.f1(x)
        # x1 = self.bn1(x1)
        
        out_cl.append(self.cl1(x1))
        out_bbx.append(self.bbx1(x1))

        x1 = self.base1(x1)
        
        x2 = self.f2(x1)

        out_cl.append(self.cl2(x2))
        out_bbx.append(self.bbx2(x2))

        x3 = self.f3(x2)

        out_cl.append(self.cl3(x3))
        out_bbx.append(self.bbx3(x3))

        x4 = self.f4(x3)

        out_cl.append(self.cl4(x4))
        out_bbx.append(self.bbx4(x4))

        x5 = self.f5(x4)
        
        out_cl.append(self.cl5(x5))
        out_bbx.append(self.bbx5(x5))

        x6 = self.f6(x5)

        out_cl.append(self.cl6(x6))
        out_bbx.append(self.bbx6(x6))

        for i in range(len(out_cl)):
            out_cl[i] = out_cl[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, self.num_classes)
            out_bbx[i] = out_bbx[i].permute(0,2,3,1).contiguous().view(out_cl[i].size(0), -1).view(out_cl[i].size(0), -1, 4)

        out_cl = torch.cat(out_cl, 1)
        out_bbx = torch.cat(out_bbx, 1)
        # res = torch.cat([out_cl, out_bbx], dim = 2)
        return out_bbx, out_cl

    def _init_weights(self, vgg_16_init=False):
    
        for module in self.layers:
            for layer in module:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
            
            if vgg_16_init:
                for module in self.vgg_layers:
                    for layer in module:
                        if isinstance(layer, nn.Conv2d):
                            nn.init.xavier_normal_(layer.weight)
                            if layer.bias is not None:
                                nn.init.constant_(layer.bias, 0)
                        elif isinstance(layer, nn.BatchNorm2d):
                            nn.init.constant_(layer.weight, 1)
                            nn.init.constant_(layer.bias, 0)


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
