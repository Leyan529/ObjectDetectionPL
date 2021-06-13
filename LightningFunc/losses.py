import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from LightningFunc.utils.YoloV4Utils import build_targets
# from LightningFunc.utils.YoloV3Utils import build_targets
# from LightningFunc.utils.YoloV2Utils import build_targets
from LightningFunc.accuracy import build_targets
from LightningFunc.accuracy import bbox_iou, iou
import numpy as np

def configure_loss(mode, iou_boxes, anchors, anch_masks, num_classes, img_size):
    if mode == 'RetinaNet':
        return RetinaNetLoss(iou_boxes, num_classes, img_size)
    if mode == 'SSD':
        return SSDLoss(iou_boxes, num_classes, img_size)
    elif mode == 'YOLOv4':
        return MultiScaleRegionLoss_v4(anchors, anch_masks, num_classes, img_size)
    elif mode == 'YOLOv3':
        return MultiScaleRegionLoss_v3(anchors, num_classes, img_size)
    elif mode == 'YOLOv2':
        return RegionLoss_v2(anchors, num_classes, img_size)

class SSDLoss(nn.Module):
    def __init__(self, iou_boxes, num_classes, img_size):
        super(SSDLoss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.default_boxes = iou_boxes  
        # SmoothL1Loss + CrossEntropyLoss

    def center_to_points(self, center_tens):
        
        if center_tens.size(0) == 0:
            return center_tens
        
        assert center_tens.dim() == 2 
        assert center_tens.size(1) == 4 

        lp = torch.clamp(center_tens[:,:2] - center_tens[:,2:]/2.0, min=0.0)
        rp = torch.clamp(center_tens[:,:2] + center_tens[:,2:]/2.0, max=1.0)

        points = torch.cat([lp, rp], 1)

        return points

    def expand_defaults_and_annotations(self, default_boxes, annotations_boxes):
        
        num_annotations = annotations_boxes.size(0)

        default_boxes = default_boxes.unsqueeze(0)
        default_boxes = default_boxes.expand(num_annotations, -1, -1)

        annotations_boxes = annotations_boxes.unsqueeze(1)
        annotations_boxes = annotations_boxes.expand_as(default_boxes)

        return default_boxes, annotations_boxes

    def match(self, default_boxes, annotations_boxes, match_thresh):        
        num_annotations = annotations_boxes.size(0)

        default_boxes_pt = self.center_to_points(default_boxes) # [#nums, #ancohors, 4]
        annotations_boxes_pt = self.center_to_points(annotations_boxes) # [#nums, #ancohors, 4]

        default_boxes_pt, annotations_boxes_pt = self.expand_defaults_and_annotations(default_boxes_pt, annotations_boxes_pt)

        # ious = bbox_iou(default_boxes_pt, annotations_boxes_pt)
        ious = iou(default_boxes_pt, annotations_boxes_pt)

        _, annotation_with_box = torch.max(ious, 1)
        annotation_inds = torch.arange(num_annotations, dtype=torch.long).to(annotation_with_box.device)
        
        ious_max, box_with_annotation = torch.max(ious, 0)
        matched_boxes_bin = (ious_max >= match_thresh)
        matched_boxes_bin[annotation_with_box] = 1
        box_with_annotation[annotation_with_box] = annotation_inds
        
        return box_with_annotation, matched_boxes_bin

    def compute_offsets(self, default_boxes, annotations_boxes, box_with_annotation_idx, use_variance=True):
        
        matched_boxes = annotations_boxes[box_with_annotation_idx]

        offset_cx = (matched_boxes[:,:2] - default_boxes[:,:2])

        if use_variance:
            offset_cx = offset_cx / (default_boxes[:,2:] * 0.1)
        else:
            offset_cx = offset_cx / default_boxes[:,2:]

        offset_wh = torch.log(matched_boxes[:,2:]/default_boxes[:,2:])

        if use_variance:
            offset_wh = offset_wh / 0.2
        
        return torch.cat([offset_cx, offset_wh], 1)

    def compute_loss(self, default_boxes, annotations_classes, annotations_boxes, predicted_classes, predicted_offsets, match_thresh=0.5, duplciate_checking=True, neg_ratio=3):
        
        if annotations_classes.size(0) > 0:
            annotations_classes = annotations_classes.long()
            box_with_annotation_idx, matched_box_bin = self.match(default_boxes, annotations_boxes, match_thresh)

            matched_box_idxs = (matched_box_bin.nonzero()).squeeze(1)
            non_matched_idxs = (matched_box_bin == 0).nonzero().squeeze(1)
            N = matched_box_idxs.size(0)

            true_offsets = self.compute_offsets(default_boxes, annotations_boxes, box_with_annotation_idx)

            regression_loss_criterion = nn.SmoothL1Loss(reduction='none')
            regression_loss = regression_loss_criterion(predicted_offsets[matched_box_idxs], true_offsets[matched_box_idxs])

            true_classifications = torch.zeros(predicted_classes.size(0), dtype=torch.long).to(predicted_classes.device)
            true_classifications[matched_box_idxs] = annotations_classes[box_with_annotation_idx[matched_box_idxs]]
        
        else:
            matched_box_idxs = torch.LongTensor([])
            non_matched_idxs = torch.arange(default_boxes.size(0))
            N = 1

            regression_loss = torch.tensor([0.0]).to(predicted_classes.device)

            true_classifications = torch.zeros(predicted_classes.size(0), dtype=torch.long).to(predicted_classes.device)
                
        classifications_loss_criterion = nn.CrossEntropyLoss(reduction='none')
        classifications_loss_total = classifications_loss_criterion(predicted_classes, true_classifications)

        positive_classifications = classifications_loss_total[matched_box_idxs]
        negative_classifications = classifications_loss_total[non_matched_idxs]

        _, hard_negative_idxs = torch.sort(classifications_loss_total[non_matched_idxs], descending=True)
        hard_negative_idxs = hard_negative_idxs.squeeze()[:N * neg_ratio]

        classifications_loss = (positive_classifications.sum() + negative_classifications[hard_negative_idxs].sum())/N
        regression_loss = regression_loss.sum()/N

        return classifications_loss, regression_loss, matched_box_idxs

    def forward(self, outputs, targets):
        predicted_offsets , predicted_classes  = outputs
        assert predicted_classes.size(0) == predicted_offsets.size(0)
        batch_size = predicted_classes.size(0)

        classification_loss = 0
        localization_loss = 0
        match_idx_viz = None

        x = targets[:, 0]
        x_unique = x.unique(sorted=True)
        lens = torch.stack([(x==x_u).sum() for x_u in x_unique])
        new_batch_size = batch_size
        for j in range(batch_size):
            current_classes = predicted_classes[j]
            current_offsets = predicted_offsets[j]
            
            # annotations_classes = targets[j][:lens[j]][:, 0] if lens[j].item() != 0 else torch.Tensor([])
            # annotations_boxes = targets[j][:lens[j]][:, 1:5] if lens[j].item() != 0 else torch.Tensor([])
            if j >= len(lens): 
                new_batch_size -= 1
                continue
            annotations_classes = targets[:lens[j]][:, 1]
            annotations_boxes = targets[:lens[j]][:, 2:]


            curr_cl_loss, curr_loc_loss, _mi = self.compute_loss(
                self.default_boxes, annotations_classes, annotations_boxes, current_classes, current_offsets)

            classification_loss += curr_cl_loss
            localization_loss += curr_loc_loss

            if j == 0:
                match_idx_viz = _mi

        localization_loss = localization_loss / new_batch_size
        classification_loss = classification_loss / new_batch_size
        total_loss = localization_loss + classification_loss
        return {"loss":total_loss, "Localization_loss":localization_loss, "Classification_loss":classification_loss}

class RetinaNetLoss(nn.Module):
    def __init__(self, iou_boxes, num_classes=20, img_size = 600):
        super(RetinaNetLoss, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.iou_boxes = iou_boxes
        # smooth_l1_loss + focal

    def _one_hot_embeding(self, labels):
        """Embeding labels to one-hot form.
        Args:
            labels(LongTensor): class labels
            num_classes(int): number of classes
        Returns:
            encoded labels, sized[N, #classes]
        """

        y = torch.eye(self.num_classes+1)  # [D, D]
        return y[labels]  # [N, D]

    def focal_loss(self, x, y):
        """Focal loss
        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
        Returns:
            (tensor): focal loss
        """

        alpha = 0.25
        gamma = 2

        t = self._one_hot_embeding(y.data.cpu())  # [N,21]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        logit = F.softmax(x)
        logit = logit.clamp(1e-7, 1.-1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logit)
        conf_loss_tmp = alpha * conf_loss_tmp * (1-logit)**gamma
        conf_loss = conf_loss_tmp.sum()
        return conf_loss

    def change_box_order(self, boxes, order):
        '''Change box order between (xmin,ymin,xmax,ymax) and (xcenter,ycenter,width,height).
        Args:
        boxes: (tensor) bounding boxes, sized [N,4].
        order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.
        Returns:
        (tensor) converted bounding boxes, sized [N,4].
        '''
        assert order in ['xyxy2xywh','xywh2xyxy']
        a = boxes[:,:2]
        b = boxes[:,2:]
        if order == 'xyxy2xywh':
            return torch.cat([(a+b)/2,b-a+1], 1)
        return torch.cat([a-b/2,a+b/2], 1)

    def box_iou(self, box1, box2, order='xyxy'):
        '''Compute the intersection over union of two set of boxes.
        The default box order is (xmin, ymin, xmax, ymax).
        Args:
        box1: (tensor) bounding boxes, sized [N,4].
        box2: (tensor) bounding boxes, sized [M,4].
        order: (str) box order, either 'xyxy' or 'xywh'.
        Return:
        (tensor) iou, sized [N,M].
        Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        '''
        if order == 'xywh':
            box1 = self.change_box_order(box1, 'xywh2xyxy')
            box2 = self.change_box_order(box2, 'xywh2xyxy')

        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
        rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

        wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
        area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
        iou = inter / (area1[:,None] + area2 - inter)
        return iou

    def forward(self, outputs, targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        ################################################################
        # split outputs and targets
        ################################################################
        loc_preds, cls_preds = outputs        
        loc_target_bsz, cls_target_bsz = [], []
        # rm_bid = []
        # orign_bsz = loc_preds.shape[0]
        """"""
        for bid in range(len(loc_preds)):
            # boxes = change_box_order(boxes, 'xyxy2xywh') # already in xywh format
            boxes = targets[targets[:,0]==bid][:, 2:] * self.img_size
            labels = targets[targets[:,0]==bid][:, 1].type(torch.LongTensor)
            ious = self.box_iou(self.iou_boxes, boxes, order='xywh')
            # if ious.shape[-1] == 0: 
            #     rm_bid.append(bid)
            #     continue
            max_ious, max_ids = ious.max(1)
            boxes = boxes[max_ids]

            loc_xy = (boxes[:,:2]-self.iou_boxes[:,:2]) / self.iou_boxes[:,2:]
            loc_wh = torch.log(boxes[:,2:]/self.iou_boxes[:,2:])
            loc_targets = torch.cat([loc_xy,loc_wh], 1)
            cls_targets = 1 + labels[max_ids]

            cls_targets[max_ious<0.5] = 0
            ignore = (max_ious>0.4) & (max_ious<0.5)  # ignore ious between [0.4,0.5]
            cls_targets[ignore] = -1  # for now just mark ignored to -1
            loc_target_bsz.append(loc_targets)
            cls_target_bsz.append(cls_targets)
        """"""
        loc_target_bsz, cls_target_bsz = torch.stack(loc_target_bsz), torch.stack(cls_target_bsz)
        loc_targets, cls_targets = loc_target_bsz, cls_target_bsz
        ################################################################

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        # loc_preds = loc_preds[list([x for x in range(orign_bsz) if x not in rm_bid])]
        # cls_preds = cls_preds[list([x for x in range(orign_bsz) if x not in rm_bid])]
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)        
        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        num_pos = max(1.0, num_pos.item())

        loss = (loc_loss+cls_loss)/num_pos
        return {"loss":loss, "Localization_loss":loc_loss/num_pos, "Classification_loss":cls_loss/num_pos}
        # return loss, loc_loss/num_pos, cls_loss/num_pos

class RegionLoss_v4(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        # super(RegionLoss, self).__init__()
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.build_targets = build_targets
        # bce_loss + mse_loss

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        # self.img_dim = 416
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None):
         # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            # x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            x.view(num_samples, 3, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h        

        # if len(targets) == 0:
        #     return output, 0
        # else:
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        bwbh = pred_boxes[..., 2:4][obj_mask]    
        shape = min(len(bwbh), len(targets))        
        wh_loss = self.mse_loss(
            torch.sqrt(torch.abs(bwbh) + 1e-32)[:shape],
            torch.sqrt(torch.abs(targets[..., 3:5]) + 1e-32)[:shape],
        )

        return total_loss, (loss_x + loss_y), wh_loss, loss_conf, loss_cls, loss_conf_obj, loss_conf_noobj
        
class MultiScaleRegionLoss_v4(RegionLoss_v4):
    def __init__(self, anchors, anch_masks, num_classes, img_dim = 416, **kwargs):
        super().__init__(anchors, num_classes, img_dim, **kwargs)
        self._anchors = anchors
        self._anch_masks = anch_masks
        # bce_loss + mse_loss

    def forward(self, output, target, seen=None):
        device = output[0].device
        loss = torch.tensor(0.0).to(device)
        loss_coord = torch.tensor(0.0).to(device)
        loss_size = torch.tensor(0.0).to(device)
        loss_conf = torch.tensor(0.0).to(device)
        loss_cls = torch.tensor(0.0).to(device)
        loss_conf_obj = torch.tensor(0.0).to(device)
        loss_conf_noobj = torch.tensor(0.0).to(device)

        # Run loss at different scales and sum resulting loss values
        for i, out in enumerate(output):    
            self.anchors = [self._anchors[mask] for mask in self._anch_masks[i]]
            self.num_anchors = len(self.anchors)

            scale_loss, \
            scale_loss_coord, \
            scale_loss_size, \
            scale_loss_conf, \
            scale_loss_cls, \
            scale_loss_conf_obj, \
            scale_loss_conf_noobj = super().forward(out, target)

            loss_coord += scale_loss_coord
            loss_size += scale_loss_size
            loss_conf += scale_loss_conf
            loss_cls += scale_loss_cls
            loss_conf_obj += scale_loss_conf_obj
            loss_conf_noobj += scale_loss_conf_noobj
            loss += scale_loss


        # Overwrite loss values with avg
        self.loss_coord = loss_coord / len(output)
        self.loss_size = loss_size / len(output)
        self.loss_conf = loss_conf / len(output)
        self.loss_cls = loss_cls / len(output)
        self.loss_tot = loss / len(output)

        self.loss_conf_obj = loss_conf_obj / len(output)
        self.loss_conf_noobj = loss_conf_noobj / len(output)

        metrics = {
            "loss": self.loss_tot,
            "Localization_loss": self.loss_coord,
            "Size_loss": self.loss_size,
            "Conf_loss": self.loss_conf,
            "Classification_loss": self.loss_cls,
            "Conf_obj_loss": self.loss_conf_obj,
            "Conf_noobj_loss": self.loss_conf_noobj        
        }
        return metrics

class RegionLoss_v3(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        # super(RegionLoss, self).__init__()
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.build_targets = build_targets

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        # self.img_dim = 416
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None):
         # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            # x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            x.view(num_samples, 3, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h        

        # if len(targets) == 0:
        #     return output, 0
        # else:
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        bwbh = pred_boxes[..., 2:4][obj_mask]    
        shape = min(len(bwbh), len(targets))        
        wh_loss = self.mse_loss(
            torch.sqrt(torch.abs(bwbh) + 1e-32)[:shape],
            torch.sqrt(torch.abs(targets[..., 3:5]) + 1e-32)[:shape],
        )

        return total_loss, (loss_x + loss_y), wh_loss, loss_conf, loss_cls, loss_conf_obj, loss_conf_noobj
        
class MultiScaleRegionLoss_v3(RegionLoss_v3):
    def __init__(self, anchors, num_classes, img_dim = 416, **kwargs):
        super().__init__(anchors, num_classes, img_dim, **kwargs)
        self._anchors = anchors
        # bce_loss + mse_loss

    def forward(self, output, target, seen=None):
        device = output[0].device
        loss = torch.tensor(0.0).to(device)
        loss_coord = torch.tensor(0.0).to(device)
        loss_size = torch.tensor(0.0).to(device)
        loss_conf = torch.tensor(0.0).to(device)
        loss_cls = torch.tensor(0.0).to(device)
        loss_conf_obj = torch.tensor(0.0).to(device)
        loss_conf_noobj = torch.tensor(0.0).to(device)

        # Run loss at different scales and sum resulting loss values
        for i, out in enumerate(output):    
            self.anchors = self._anchors[i]
            self.num_anchors = len(self.anchors)

            scale_loss, \
            scale_loss_coord, \
            scale_loss_size, \
            scale_loss_conf, \
            scale_loss_cls, \
            scale_loss_conf_obj, \
            scale_loss_conf_noobj = super().forward(out, target)

            loss_coord += scale_loss_coord
            loss_size += scale_loss_size
            loss_conf += scale_loss_conf
            loss_cls += scale_loss_cls
            loss_conf_obj += scale_loss_conf_obj
            loss_conf_noobj += scale_loss_conf_noobj
            loss += scale_loss


        # Overwrite loss values with avg
        self.loss_coord = loss_coord / len(output)
        self.loss_size = loss_size / len(output)
        self.loss_conf = loss_conf / len(output)
        self.loss_cls = loss_cls / len(output)
        self.loss_tot = loss / len(output)

        self.loss_conf_obj = loss_conf_obj / len(output)
        self.loss_conf_noobj = loss_conf_noobj / len(output)

        metrics = {
            "loss": self.loss_tot,
            "Localization_loss": self.loss_coord,
            "Size_loss": self.loss_size,
            "Conf_loss": self.loss_conf,
            "Classification_loss": self.loss_cls,
            "Conf_obj_loss": self.loss_conf_obj,
            "Conf_noobj_loss": self.loss_conf_noobj        
        }
        return metrics

class RegionLoss_v2(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=416):
        # super(RegionLoss, self).__init__()
        super().__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.build_targets = build_targets
        # bce_loss + mse_loss

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        # self.img_dim = 416
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None):
         # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h        

        # if len(targets) == 0:
        #     return output, 0
        # else:
        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = self.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
        loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
        loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
        total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        bwbh = pred_boxes[..., 2:4][obj_mask]    
        shape = min(len(bwbh), len(targets))        
        wh_loss = self.mse_loss(
            torch.sqrt(torch.abs(bwbh) + 1e-32)[:shape],
            torch.sqrt(torch.abs(targets[..., 3:5]) + 1e-32)[:shape],
        )

        metrics = {
            "loss": total_loss,
            "Localization_loss": (loss_x + loss_y),
            "Size_loss": wh_loss,
            "Conf_loss": loss_conf,
            "Classification_loss": loss_cls,
            "Conf_obj_loss": loss_conf_obj,
            "Conf_noobj_loss": loss_conf_noobj        
        }
        return metrics

  


