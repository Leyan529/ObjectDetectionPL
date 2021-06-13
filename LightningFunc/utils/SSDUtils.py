import torch
import numpy as np
import itertools

def get_dboxes(smin=0.07, smax=0.9, ars=[1, 2, (1/2.0), 3, (1/3.0)], fks=[38, 19, 10, 5, 3, 1], num_boxes=[3, 5, 5, 5, 3, 3]):
    m = len(fks)
    sks = [round(smin + (((smax-smin)/(m-1)) * (k-1)), 2) for k in range(1, m + 1)]

    boxes = []
    for k, feat_k in enumerate(fks):
        for i, j in itertools.product(range(feat_k), range(feat_k)):

            cx = (i + 0.5)/feat_k
            cy = (j + 0.5)/feat_k

            w = h = np.sqrt(sks[k] * sks[min(k+1, len(sks) - 1)])

            boxes.append([cx, cy, w, h])

            sk = sks[k]
            for ar in ars[:num_boxes[k]]:
                w = sk * np.sqrt(ar)
                h = sk / np.sqrt(ar)
                boxes.append([cx, cy, w, h])

    boxes = torch.tensor(boxes).float()
    return torch.clamp(boxes, max=1.0)
