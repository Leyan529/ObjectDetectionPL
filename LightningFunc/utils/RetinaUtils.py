import torch
import math
# RetinaNet
# -----------------------------------------------------------------------------------------------------------------------------------------

def get_anchor_boxes(input_size):
    '''Compute anchor boxes for each feature map.
    Args:
        input_size: (tensor) model input size of (w,h).
    Returns:
        boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                    where #anchors = fmw * fmh * #anchors_per_cell
    '''
    anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]
    aspect_ratios = [1/2., 1/1., 2/1.]
    scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
    anchor_wh = get_anchor_wh(anchor_areas, aspect_ratios, scale_ratios)
    num_fms = len(anchor_areas)
    fm_sizes = [(input_size/pow(2.,i+3)).ceil() for i in range(num_fms)]  # p3 -> p7 feature map sizes

    boxes = []
    for i in range(num_fms):
        fm_size = fm_sizes[i]
        grid_size = input_size / fm_size
        fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
        xy = (meshgrid(fm_w,fm_h) + 0.5)  # [fm_h*fm_w, 2]
        xy = (xy*grid_size).view(fm_h,fm_w,1,2).expand(fm_h,fm_w,9,2)
        wh = anchor_wh[i].view(1,1,9,2).expand(fm_h,fm_w,9,2)
        box = torch.cat([xy,wh], 3)  # [x,y,w,h]
        boxes.append(box.view(-1,4))
    return torch.cat(boxes, 0)

def meshgrid(x, y, row_major=True):
    '''Return meshgrid in range x & y.
    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      row_major: (bool) row major or column major.
    Returns:
      (tensor) meshgrid, sized [x*y,2]
    Example:
    >> meshgrid(3,2)
    0  0
    1  0
    2  0
    0  1
    1  1
    2  1
    [torch.FloatTensor of size 6x2]
    >> meshgrid(3,2,row_major=False)
    0  0
    0  1
    0  2
    1  0
    1  1
    1  2
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x,dtype=torch.float)
    b = torch.arange(0,y,dtype=torch.float)
    xx = a.repeat(y).view(-1,1)
    yy = b.view(-1,1).repeat(1,x).view(-1,1)
    return torch.cat([xx,yy],1) if row_major else torch.cat([yy,xx],1)

def get_anchor_wh(anchor_areas, aspect_ratios, scale_ratios):
    '''Compute anchor width and height for each feature map.
    Returns:
        anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
    '''
    anchor_wh = []
    for s in anchor_areas:
        for ar in aspect_ratios:  # w/h = ar
            h = math.sqrt(s/ar)
            w = ar * h
            for sr in scale_ratios:  # scale
                anchor_h = h*sr
                anchor_w = w*sr
                anchor_wh.append([anchor_w, anchor_h])
    num_fms = len(anchor_areas)
    return torch.Tensor(anchor_wh).view(num_fms, -1, 2)


