import torch
from torch import nn
from collections import OrderedDict, Iterable
import math

from LightningFunc.utils.YoloV3Utils import *
from LightningFunc.accuracy import xywh2xyxy, bbox_iou

import pytorch_lightning as pl

from LightningFunc.step import *
from LightningFunc.accuracy import *
from LightningFunc.optimizer import *
from LightningFunc.lightningUtils import *
from LightningFunc.losses import configure_loss
import pickle


class YOLOv3(pl.LightningModule):
    """ Yolo v3 implementation :cite:`yolo_v3`.
    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 3D list with anchor values; Default **Yolo v3 anchors**
    Attributes:
        self.stride: Subsampling factors of the network (input dimensions should be a multiple of these numbers)
        self.remap_darknet53: Remapping rules for weights from the `~lightnet.models.Darknet53` model.
    Note:
        Unlike YoloV2, the anchors here are defined as multiples of the input dimensions and not as a multiple of the output dimensions!
        The anchor list also has one more dimension than the one from YoloV2, in order to differentiate which anchors belong to which stride.
    Warning:
        The :class:`~lightnet.network.loss.MultiScaleRegionLoss` and :class:`~lightnet.data.transform.GetMultiScaleBoundingBoxes`
        do not implement the overlapping class labels of the original implementation.
        Your weight files from darknet will thus not have the same accuracies as in darknet itself.
    """
    stride = (32, 16, 8)
    remap_darknet53 = [
        (r'^layers.([a-w]_)',   r'extractor.\1'),   # Residual layers
        (r'^layers.(\d_)',      r'extractor.\1'),   # layers 1, 2, 5
        (r'^layers.([124]\d_)', r'extractor.\1'),   # layers 10, 27, 44
    ]
    sample_anchors=[[(116, 90), (156, 198), (373, 326)], [(30, 61), (62, 45), (59, 119)], [(10, 13), (16, 30), (33, 23)]]
    img_size = 416
    grid_size = 0
    ignore_thres = 0.5
    colors = pickle.load(open("dataset//pallete", "rb"))   
    anch_masks = None

    def __init__(self, classes, data_name):
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)        

        self.anchors = []   # YoloV3 defines anchors as a multiple of the input dimensions of the network as opposed to the output dimensions
        for i, s in enumerate(self.stride):
            self.anchors.append([(a[0] / s, a[1] / s) for a in self.sample_anchors[i]])
            
        self.__build_model()
        self.__build_func(YOLOv3)   
        self.sample = (1, 3, 416, 416)
        self.sampleImg=torch.rand(self.sample).cuda()

        self.criterion = configure_loss('YOLOv3', None, self.anchors, None, self.num_classes, self.img_size)

        self.checkname = self.backbone
        self.data_name = data_name
        self.dir = os.path.join("log_dir", self.data_name ,self.checkname)

    def __build_model(self):
        input_channels = 3

        
        # Network
        self.extractor = SelectiveSequential(
            ['k_residual', 's_residual'],
            OrderedDict([
                ('1_convbatch',         Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_convbatch',         Conv2dBatchReLU(32, 64, 3, 2, 1)),
                ('a_residual',          Residual(OrderedDict([
                    ('3_convbatch',     Conv2dBatchReLU(64, 32, 1, 1, 0)),
                    ('4_convbatch',     Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ]))),
                ('5_convbatch',         Conv2dBatchReLU(64, 128, 3, 2, 1)),
                ('b_residual',          Residual(OrderedDict([
                    ('6_convbatch',     Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('7_convbatch',     Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('c_residual',          Residual(OrderedDict([
                    ('8_convbatch',     Conv2dBatchReLU(128, 64, 1, 1, 0)),
                    ('9_convbatch',     Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ]))),
                ('10_convbatch',        Conv2dBatchReLU(128, 256, 3, 2, 1)),
                ('d_residual',          Residual(OrderedDict([
                    ('11_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('12_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('e_residual',          Residual(OrderedDict([
                    ('13_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('14_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('f_residual',          Residual(OrderedDict([
                    ('15_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('16_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('g_residual',          Residual(OrderedDict([
                    ('17_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('18_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('h_residual',          Residual(OrderedDict([
                    ('19_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('20_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('i_residual',          Residual(OrderedDict([
                    ('21_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('22_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('j_residual',          Residual(OrderedDict([
                    ('23_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('24_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('k_residual',          Residual(OrderedDict([
                    ('25_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('26_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ]))),
                ('27_convbatch',        Conv2dBatchReLU(256, 512, 3, 2, 1)),
                ('l_residual',          Residual(OrderedDict([
                    ('28_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('29_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('m_residual',          Residual(OrderedDict([
                    ('30_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('31_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('n_residual',          Residual(OrderedDict([
                    ('32_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('33_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('o_residual',          Residual(OrderedDict([
                    ('34_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('35_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('p_residual',          Residual(OrderedDict([
                    ('36_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('37_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('q_residual',          Residual(OrderedDict([
                    ('38_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('39_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('r_residual',          Residual(OrderedDict([
                    ('40_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('41_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('s_residual',          Residual(OrderedDict([
                    ('42_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('43_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ]))),
                ('44_convbatch',        Conv2dBatchReLU(512, 1024, 3, 2, 1)),
                ('t_residual',          Residual(OrderedDict([
                    ('45_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('46_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('u_residual',          Residual(OrderedDict([
                    ('47_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('48_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('v_residual',          Residual(OrderedDict([
                    ('49_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('50_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
                ('w_residual',          Residual(OrderedDict([
                    ('51_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('52_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ]))),
            ]),
        )

        self.detector = nn.ModuleList([
            # Sequence 0 : input = extractor
            SelectiveSequential(
                ['57_convbatch'],
                OrderedDict([
                    ('53_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('54_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('55_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('56_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('57_convbatch',    Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                    ('58_convbatch',    Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                    ('59_conv',         nn.Conv2d(1024, len(self.anchors[0])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 1 : input = 57_convbatch
            nn.Sequential(
                OrderedDict([
                    ('60_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('61_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 2 : input = 61_upsample and s_residual
            SelectiveSequential(
                ['66_convbatch'],
                OrderedDict([
                    ('62_convbatch',    Conv2dBatchReLU(256+512, 256, 1, 1, 0)),
                    ('63_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('64_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('65_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('66_convbatch',    Conv2dBatchReLU(512, 256, 1, 1, 0)),
                    ('67_convbatch',    Conv2dBatchReLU(256, 512, 3, 1, 1)),
                    ('68_conv',         nn.Conv2d(512, len(self.anchors[1])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),

            # Sequence 3 : input = 66_convbatch
            nn.Sequential(
                OrderedDict([
                    ('69_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('70_upsample',     nn.Upsample(scale_factor=2, mode='nearest')),
                ])
            ),

            # Sequence 4 : input = 70_upsample and k_residual
            nn.Sequential(
                OrderedDict([
                    ('71_convbatch',    Conv2dBatchReLU(128+256, 128, 1, 1, 0)),
                    ('72_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('73_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('74_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('75_convbatch',    Conv2dBatchReLU(256, 128, 1, 1, 0)),
                    ('76_convbatch',    Conv2dBatchReLU(128, 256, 3, 1, 1)),
                    ('77_conv',         nn.Conv2d(256, len(self.anchors[2])*(5+self.num_classes), 1, 1, 0)),
                ])
            ),
        ])
    
    def __build_func(self, obj):
        """Define model layers & loss."""

        self.backbone = "YOLOv3"
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
        out = [None, None, None]

        # Feature extractor
        x, inter_features = self.extractor(x)

        # detector 0
        out[0], x = self.detector[0](x)

        # detector 1
        x = self.detector[1](x)
        out[1], x = self.detector[2](torch.cat((x, inter_features['s_residual']), 1))

        # detector 2
        x = self.detector[3](x)
        out[2] = self.detector[4](torch.cat((x, inter_features['k_residual']), 1))

        return out

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
            # [bsz, 3*(13+6),52*52]
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
            prediction = prediction.view(num_samples, -1 , answers) # torch.Size([1, 507, 25])
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

       
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
    over_b = torch.sum(torch.Tensor([d>= obj_mask.shape[0] for d in b]))
    over_n = torch.sum(torch.Tensor([d>= obj_mask.shape[1] for d in best_n]))
    over_gj = torch.sum(torch.Tensor([d>= obj_mask.shape[2] for d in gj]))
    over_gi = torch.sum(torch.Tensor([d>= obj_mask.shape[3] for d in gi]))
    if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() == 0:    
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
        if i >= len(b): continue
        if i >= len(gj): continue
        if i >= len(gi): continue
        if b[i] >= noobj_mask.shape[0]: continue
        if gj[i] >= noobj_mask.shape[2]: continue
        if gi[i] >= noobj_mask.shape[3]: continue
        
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    over_b = torch.sum(torch.Tensor([d>= tx.shape[0] for d in b]))
    over_n = torch.sum(torch.Tensor([d>= tx.shape[1] for d in best_n]))
    over_gj = torch.sum(torch.Tensor([d>= tx.shape[2] for d in gj]))
    over_gi = torch.sum(torch.Tensor([d>= tx.shape[3] for d in gi]))
    over_labels = torch.sum(torch.Tensor([d>= tcls.shape[4] for d in target_labels]))
    # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
    if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() + over_labels.item() == 0:
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()
        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
        # One-hot encoding of label
        tcls[b, best_n, gj, gi, target_labels] = 1
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

# def bbox_wh_iou(wh1, wh2):
#     wh2 = wh2.t()
#     w1, h1 = wh1[0], wh1[1]
#     w2, h2 = wh2[0], wh2[1]
#     inter_area = torch.min(w1, w2) * torch.min(h1, h2)
#     union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
#     return inter_area / union_area


# def bbox_iou(box1, box2, x1y1x2y2=True):
#     """
#     Returns the IoU of two bounding boxes
#     """
#     if not x1y1x2y2:
#         # Transform from center and width to exact coordinates
#         b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#         b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#         b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#         b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
#     else:
#         # Get the coordinates of bounding boxes
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

#     # get the corrdinates of the intersection rectangle
#     inter_rect_x1 = torch.max(b1_x1, b2_x1)
#     inter_rect_y1 = torch.max(b1_y1, b2_y1)
#     inter_rect_x2 = torch.min(b1_x2, b2_x2)
#     inter_rect_y2 = torch.min(b1_y2, b2_y2)
#     # Intersection area
#     inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
#         inter_rect_y2 - inter_rect_y1 + 1, min=0
#     )
#     # Union Area
#     b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
#     b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

#     return iou

# def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
#     """
#     Removes detections with lower object confidence score than 'conf_thres' and performs
#     Non-Maximum Suppression to further filter detections.
#     Returns detections with shape:
#         (x1, y1, x2, y2, object_conf, class_score, class_pred)
#     """

#     # From (center x, center y, width, height) to (x1, y1, x2, y2)
#     prediction[..., :4] = xywh2xyxy(prediction[..., :4])
#     output = [None for _ in range(len(prediction))]
#     for image_i, image_pred in enumerate(prediction):
#         # Filter out confidence scores below threshold
#         image_pred = image_pred[image_pred[:, 4] >= conf_thres]
#         # If none are remaining => process next image
#         if not image_pred.size(0):
#             continue
#         # Object confidence times class confidence
#         score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
#         # Sort by it
#         image_pred = image_pred[(-score).argsort()]
#         class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
#         detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
#         # Perform non-maximum suppression
#         keep_boxes = []
#         while detections.size(0):
#             large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
#             label_match = detections[0, -1] == detections[:, -1]
#             # Indices of boxes with lower confidence scores, large IOUs and matching labels
#             invalid = large_overlap & label_match
#             weights = detections[invalid, 4:5]
#             # Merge overlapping bboxes by order of confidence
#             detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
#             keep_boxes += [detections[0]]
#             detections = detections[~invalid]
#         if keep_boxes:
#             output[image_i] = torch.stack(keep_boxes)

#     return output

# def to_cpu(tensor):
#     return tensor.detach().cpu()

if __name__ == "__main__":  
    net = YOLOv3(num_classes=20)
    print(net)