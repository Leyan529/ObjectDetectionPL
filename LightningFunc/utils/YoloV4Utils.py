import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


# def xywh2xyxy(x):
#     y = x.new(x.shape)
#     y[..., 0] = x[..., 0] - x[..., 2] / 2
#     y[..., 1] = x[..., 1] - x[..., 3] / 2
#     y[..., 2] = x[..., 0] + x[..., 2] / 2
#     y[..., 3] = x[..., 1] + x[..., 3] / 2
#     return y



def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]        

def yolo_forward_dynamic(output, conf_thresh, num_classes, anchors, num_anchors, scale_x_y, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    # batch = output.size(0)
    # H = output.size(2)
    # W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3))

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(output.size(0), num_anchors, num_classes, output.size(2) * output.size(3))
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(output.size(0), num_anchors * output.size(2) * output.size(3), num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy) * scale_x_y - 0.5 * (scale_x_y - 1)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.sigmoid(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(3) - 1, output.size(3)), axis=0).repeat(output.size(2), 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, output.size(2) - 1, output.size(2)), axis=1).repeat(output.size(3), 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii : ii + 1] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1 : ii + 2] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii : ii + 1] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1 : ii + 2] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # Shape: [batch, 2 * num_anchors, H, W]
    bx_bw = torch.cat((bx, bw), dim=1)
    # Shape: [batch, 2 * num_anchors, H, W]
    by_bh = torch.cat((by, bh), dim=1)

    # normalize coordinates to [0, 1]
    bx_bw /= output.size(3)
    by_bh /= output.size(2)

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx_bw[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    by = by_bh[:, :num_anchors].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bw = bx_bw[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    bh = by_bh[:, num_anchors:].view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)

    bx1 = bx - bw * 0.5
    by1 = by - bh * 0.5
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Shape: [batch, num_anchors * h * w, 4] -> [batch, num_anchors * h * w, 1, 4]
    boxes = torch.cat((bx1, by1, bx2, by2), dim=2).view(output.size(0), num_anchors * output.size(2) * output.size(3), 1, 4)
    # boxes = boxes.repeat(1, 1, num_classes, 1)

    # boxes:     [batch, num_anchors * H * W, 1, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(output.size(0), num_anchors * output.size(2) * output.size(3), 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 1, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x

class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8

class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5

class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6

       
# def build_targets_v4(pred_boxes, pred_cls, target, anchors, ignore_thres):
    
#     ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
#     FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

#     nB = pred_boxes.size(0)
#     nA = pred_boxes.size(1)
#     nC = pred_cls.size(-1)
#     nG = pred_boxes.size(2)

#     # Output tensors
#     obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
#     noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
#     class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
#     iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
#     tx = FloatTensor(nB, nA, nG, nG).fill_(0)
#     ty = FloatTensor(nB, nA, nG, nG).fill_(0)
#     tw = FloatTensor(nB, nA, nG, nG).fill_(0)
#     th = FloatTensor(nB, nA, nG, nG).fill_(0)
#     tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

#     # Convert to position relative to box
#     target_boxes = target[:, 2:6] * nG
#     gxy = target_boxes[:, :2]
#     gwh = target_boxes[:, 2:]
#     # Get anchors with best iou
#     ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
#     best_ious, best_n = ious.max(0)
#     # Separate target values
#     b, target_labels = target[:, :2].long().t()
#     gx, gy = gxy.t()
#     gw, gh = gwh.t()
#     gi, gj = gxy.long().t()
#     # Set masks
#     # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
#     over_b = torch.sum(torch.Tensor([d>= obj_mask.shape[0] for d in b]))
#     over_n = torch.sum(torch.Tensor([d>= obj_mask.shape[1] for d in best_n]))
#     over_gj = torch.sum(torch.Tensor([d>= obj_mask.shape[2] for d in gj]))
#     over_gi = torch.sum(torch.Tensor([d>= obj_mask.shape[3] for d in gi]))
#     if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() == 0:    
#         obj_mask[b, best_n, gj, gi] = 1
#         noobj_mask[b, best_n, gj, gi] = 0

#     # Set noobj mask to zero where iou exceeds ignore threshold
#     for i, anchor_ious in enumerate(ious.t()):
#         # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
#         if i >= len(b): continue
#         if i >= len(gj): continue
#         if i >= len(gi): continue
#         if b[i] >= noobj_mask.shape[0]: continue
#         if gj[i] >= noobj_mask.shape[2]: continue
#         if gi[i] >= noobj_mask.shape[3]: continue
        
#         noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

#     # Coordinates
#     over_b = torch.sum(torch.Tensor([d>= tx.shape[0] for d in b]))
#     over_n = torch.sum(torch.Tensor([d>= tx.shape[1] for d in best_n]))
#     over_gj = torch.sum(torch.Tensor([d>= tx.shape[2] for d in gj]))
#     over_gi = torch.sum(torch.Tensor([d>= tx.shape[3] for d in gi]))
#     over_labels = torch.sum(torch.Tensor([d>= tcls.shape[4] for d in target_labels]))
#     # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
#     if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() + over_labels.item() == 0:
#         tx[b, best_n, gj, gi] = gx - gx.floor()
#         ty[b, best_n, gj, gi] = gy - gy.floor()
#         # Width and height
#         tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
#         th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
#         # One-hot encoding of label
#         tcls[b, best_n, gj, gi, target_labels] = 1
#         # Compute label correctness and iou at best anchor
#         class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
#         iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

#     tconf = obj_mask.float()
#     return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

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

# def build_targets_v4_xxx(pred_boxes, pred_cls, target, anchors, ignore_thres):
       
#     ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
#     FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

#     nB = pred_boxes.size(0)
#     nA = pred_boxes.size(1)
#     nC = pred_cls.size(-1)
#     nG = pred_boxes.size(2)

#     # Output tensors
#     obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
#     noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
#     class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
#     iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
#     tx = FloatTensor(nB, nA, nG, nG).fill_(0)
#     ty = FloatTensor(nB, nA, nG, nG).fill_(0)
#     tw = FloatTensor(nB, nA, nG, nG).fill_(0)
#     th = FloatTensor(nB, nA, nG, nG).fill_(0)
#     tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

#     # Convert to position relative to box
#     target_boxes = target[:, 2:6] * nG
#     gxy = target_boxes[:, :2]
#     gwh = target_boxes[:, 2:]
#     # Get anchors with best iou
#     ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
#     best_ious, best_n = ious.max(0)
#     # Separate target values
#     b, target_labels = target[:, :2].long().t()
#     gx, gy = gxy.t()
#     gw, gh = gwh.t()
#     gi, gj = gxy.long().t()
#     # Set masks
#     # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
#     over_b = torch.sum(torch.Tensor([d>= obj_mask.shape[0] for d in b]))
#     over_n = torch.sum(torch.Tensor([d>= obj_mask.shape[1] for d in best_n]))
#     over_gj = torch.sum(torch.Tensor([d>= obj_mask.shape[2] for d in gj]))
#     over_gi = torch.sum(torch.Tensor([d>= obj_mask.shape[3] for d in gi]))
#     # if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() == 0:    
#     obj_mask[b, best_n, gj, gi] = 1
#     noobj_mask[b, best_n, gj, gi] = 0

#     # Set noobj mask to zero where iou exceeds ignore threshold
#     for i, anchor_ious in enumerate(ious.t()):
#         # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
#         # if i >= len(b): continue
#         # if i >= len(gj): continue
#         # if i >= len(gi): continue
#         # if b[i] >= noobj_mask.shape[0]: continue
#         # if gj[i] >= noobj_mask.shape[2]: continue
#         # if gi[i] >= noobj_mask.shape[3]: continue
        
#         noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

#     # Coordinates
#     over_b = torch.sum(torch.Tensor([d>= tx.shape[0] for d in b]))
#     over_n = torch.sum(torch.Tensor([d>= tx.shape[1] for d in best_n]))
#     over_gj = torch.sum(torch.Tensor([d>= tx.shape[2] for d in gj]))
#     over_gi = torch.sum(torch.Tensor([d>= tx.shape[3] for d in gi]))
#     over_labels = torch.sum(torch.Tensor([d>= tcls.shape[4] for d in target_labels]))
#     # CUDA error: device-side assert triggered (訓練資料的 Label 中是否存在著 -1) -> loss = nan
#     # if over_b.item() + over_n.item() + over_gj.item() + over_gi.item() + over_labels.item() == 0:
#     tx[b, best_n, gj, gi] = gx - gx.floor()
#     ty[b, best_n, gj, gi] = gy - gy.floor()
#     # Width and height
#     tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
#     th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
#     # One-hot encoding of label
#     tcls[b, best_n, gj, gi, target_labels] = 1
#     # Compute label correctness and iou at best anchor
#     class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
#     iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

#     tconf = obj_mask.float()
#     return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf    

