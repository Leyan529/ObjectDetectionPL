import numpy as np
import torch
import cv2
import math

def iou(tens1, tens2):
    
    assert tens1.size() == tens2.size()

    squeeze = False
    if tens1.dim() == 2 and tens2.dim() == 2:
        squeeze = True
        tens1 = tens1.unsqueeze(0)
        tens2 = tens2.unsqueeze(0)
    
    assert tens1.dim() == 3 
    assert tens1.size(-1) == 4 and tens2.size(-1) == 4

    maxs = torch.max(tens1[:,:,:2], tens2[:,:,:2])
    mins = torch.min(tens1[:,:,2:], tens2[:,:,2:])

    diff = torch.clamp(mins - maxs, min=0.0)

    intersection = diff[:,:,0] * diff[:,:,1]

    diff1 = torch.clamp(tens1[:,:,2:] - tens1[:,:,:2], min=0.0)
    area1 = diff1[:,:,0] * diff1[:,:,1]

    diff2 = torch.clamp(tens2[:,:,2:] - tens2[:,:,:2], min=0.0)
    area2 = diff2[:,:,0] * diff2[:,:,1]

    iou = intersection/(area1 + area2 - intersection)

    if squeeze:
        iou = iou.squeeze(0)
    
    return iou

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union = (b1_area + b2_area - inter_area + 1e-16)

    iou = inter_area / union
    return iou

def bbox_iou_v5(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box2 = box2.t()

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    # TPs, confs, preds = [], [], []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores.cpu().data.numpy(), pred_labels.cpu().data.numpy()])
    return batch_metrics

def mark_target(self, img, targets, index):
    # img = cv2.UMat(img).get()
    for target in targets:
        target = target.numpy()
        if target[0] == index:
            box = target[2:]
            cls_id = int(target[1])
            color = self.colors[cls_id]
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            xmax += xmin
            ymax += ymin
            # img = np.array(img, dtype=np.uint8) 
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(self.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            cv2.putText(
                img, self.classes[cls_id],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
            # print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax))  
    
    # cv2.imshow('win', img)
    # cv2.waitKey()
    return img
   
        
def mark_pred(self, pred_img, pred_boxes):
    # pred_img = np.array(pred_img.permute(1, 2, 0).cpu()*255, dtype=np.uint8) # Re multiply
    # for pred_boxes in suppress_output:
    if type(None) == type(pred_boxes): return pred_img 
    for target in pred_boxes:
        target = target.cpu().numpy() # (x1, y1, x2, y2, object_conf, class_score, class_pred)
        box = target[:4]
        cls_id = int(target[6])
        color = self.colors[cls_id]
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        xmax += xmin
        ymax += ymin
        cv2.rectangle(pred_img, (xmin, ymin), (xmax, ymax), color, 2)
        text_size = cv2.getTextSize(self.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(pred_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
        cv2.putText(
            pred_img, self.classes[cls_id],
            (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            (255, 255, 255), 1)
        # print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax)) 

    # cv2.imshow('win', pred_img)
    # cv2.waitKey()
    return pred_img

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    # for c in tqdm(unique_classes, desc="Computing AP"):
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

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

def get_yolo_statistics(self, output, target):
    # _anchors = self.anchors
    # _anch_masks = self._anch_masks
    batch_metrics = {}
    if type(output) != list: output = [output]
    for i, x in enumerate(output):    
        if self.anch_masks!= None: # yolo_v4
            anchors = [self.anchors[mask] for mask in self.anch_masks[i]]
        # elif type(self.anchors[i][0]) == int:
        #     _anchors = self.anchors[i]
        #     anchors = [[_anchors[idx*2] , _anchors[idx*2 + 1]] for idx in range(len(_anchors)//2)]
        else:
            if len(self.anchors) == 3:
                anchors = self.anchors[i] # yolo_v3
            else:
                anchors = self.anchors # yolo_v2

        self.num_anchors = len(anchors)

        '''scale_acc_result = super().forward(out, target)'''
        num_samples = x.size(0)
        grid_size = x.size(2)
        # torch.Size([1, 75, 13, 13])
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

        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = torch.cuda.FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h        

        iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=target,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres,
        )
        obj_mask = obj_mask.bool()
        noobj_mask = noobj_mask.bool()

        # Metrics
        cls_acc = 100 * class_mask[obj_mask].mean()
        conf_obj = pred_conf[obj_mask].mean()
        conf_noobj = pred_conf[noobj_mask].mean()
        conf50 = (pred_conf > 0.5).float()
        iou50 = (iou_scores > 0.5).float()
        iou75 = (iou_scores > 0.75).float()
        detected_mask = conf50 * class_mask * tconf
        precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
        recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
        recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        batch_metrics[grid_size] = [cls_acc.cpu().data.numpy(), recall50.cpu().data.numpy(), recall75.cpu().data.numpy(), \
                                    precision.cpu().data.numpy(), conf_obj.cpu().data.numpy(), conf_noobj.cpu().data.numpy(), output.cpu()]
    return batch_metrics

def build_targets_v5(p, targets, anchors, nl, na):
    nt = targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets
    at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
    anchor_t= 4.0  # anchor-multiple threshold

    style = 'rect4'
    for i in range(nl):
        _anchors = anchors[i]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        a, t, offsets = [], targets * gain, 0
        if nt:
            r = t[None, :, 4:6] / _anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < anchor_t  # compare
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

            # overlaps
            gxy = t[:, 2:4]  # grid xy
            z = torch.zeros_like(gxy)
            if style == 'rect2':
                g = 0.2  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                a, t = torch.cat((a, a[j], a[k]), 0), torch.cat((t, t[j], t[k]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1]), 0) * g

            elif style == 'rect4':
                g = 0.5  # offset
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(_anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch