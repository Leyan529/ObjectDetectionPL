import torch
# from torch._C import dtype, float32
# from torch._C import dtype
# from torchvision.utils import save_image

from LightningFunc.lightningUtils import *
from LightningFunc.optimizer import get_lr
import numpy as np
import cv2
from copy import deepcopy
from LightningFunc.accuracy import get_batch_statistics, ap_per_class


def training_step(self, batch, batch_idx):
    # training_step defined the train loop.
    # It is independent of forward
    x, y = batch
    out = self.forward(x)
    result_dict = self.criterion(out, y) 

    for key, val in result_dict.items():    
        if key == "loss": key = "total"
        self.logger.experiment.add_scalars("Loss/%s"%(key), {"Train":val}, self.global_step)        
    
    return result_dict


    

def training_epoch_end(self, outputs): # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
    keys = outputs[0].keys()
    for key in keys:
        avg_tensor = torch.stack([x[key] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Epoch/%s"%(key), {"Train":avg_tensor}, self.current_epoch)

    if(self.current_epoch==1):    
        self.logger.experiment.add_graph(self, self.sampleImg)

    # iterating through all parameters
    for name,params in self.named_parameters():       
        self.logger.experiment.add_histogram(name,params,self.current_epoch)
    
def validation_step(self, batch, batch_idx):
    x, y = batch
    out = self.forward(x)
    result_dict = self.criterion(out, y) 

    self.log_dict({"val_loss":result_dict["loss"]}, logger=True, on_epoch=True)
    
    return result_dict
            


def validation_epoch_end(self, outputs): # 在Validation的一個Epoch結束後，計算平均的Loss及Acc.
    keys = outputs[0].keys()
    for key in keys:
        avg_tensor = torch.stack([x[key] for x in outputs]).mean()
        self.logger.experiment.add_scalars("Epoch/%s"%(key), {"Val":avg_tensor}, self.current_epoch)
        if key == "loss": key = "total"
        self.logger.experiment.add_scalars("Loss/%s"%(key), {"Val":avg_tensor}, self.global_step)

    self.write_Best_model_path()  

def test_step(self, batch, batch_idx): #定義 Test 階段
    self.inference = True
    x, y = batch
    out = self.forward(x)

    # non_max_suppression
    suppress_output = self.non_max_suppression(out) 

    vis_list = []
    for idx, img in enumerate(x):
        img = x[idx].cpu()
        img = np.array(img*255, dtype=np.uint8) 
        img = np.transpose(img, (1, 2, 0))
        img = cv2.UMat(img).get()
        pred_img = deepcopy(img)

        # target_img
        y[:, 2:] *= self.img_size
        target_img = self.mark_target(img, y.cpu(), idx)

        # pred_img        
        pred_img = self.mark_pred(pred_img, suppress_output[idx])

        # merge
        vis = np.concatenate((target_img, pred_img), axis=1)
        vis_list.append(vis)
        # cv2.imshow('win', vis)
        # cv2.waitKey()
    
    if self.checkname in ["RetinaNet", "SSD", "YOLOv5"]:
        # List of tuples (TP, confs, pred)
        sample_metric = get_batch_statistics(suppress_output, y.cuda(), iou_threshold=0.5) 
        label = y[:, 1].tolist()
        return {"sample_metric": sample_metric, "label": label, "vis":torch.from_numpy(np.array(vis_list[0]))/255}
    else:
        sample_metric = self.get_yolo_statistics(out, y) 
        return {"sample_metric": sample_metric, "vis":torch.from_numpy(np.array(vis_list[0]))/255}
    
def test_epoch_end(self, outputs): 
    vis_list = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    if self.checkname in ["RetinaNet", "SSD", "YOLOv5"]:
        labels = []    
        for x in outputs:
            sample_metrics += x['sample_metric']
            labels += x["label"]
            vis_list += x["vis"].unsqueeze(0)

        # Concatenate sample statistics
        for x in list(zip(*sample_metrics)):
            np.concatenate(x, 0) 
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        # AP: average precision
        self.logger.experiment.add_scalars("Test/precision", {"": precision.mean()}, self.current_epoch)  
        self.logger.experiment.add_scalars("Test/recall", {"": recall.mean() }, self.current_epoch)  
        self.logger.experiment.add_scalars("Test/AP", {"":AP.mean()}, self.current_epoch)  
        self.logger.experiment.add_scalars("Test/f1", {"": f1.mean()}, self.current_epoch)  
        for i, c in enumerate(ap_class):
            self.logger.experiment.add_scalars("Test/ap_class", {self.classes[c]: AP[i]}, self.current_epoch)  

        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({self.classes[c]}) - AP: {AP[i]}")

        print(f"mAP: {AP.mean()}")
    else:        
        scale = list(outputs[0]['sample_metric'].keys())
        # metrics = ['cls_acc', 'recall50', 'recall75', 'precision', 'conf_obj', 'conf_noobj', 'output']
        for s in scale:
            sample_metrics = []
            for x in outputs:  
                item = x['sample_metric'][s]     
                sample_metrics += [item]  
                if s == 52:
                    vis_list += x["vis"].unsqueeze(0)
        
            cls_accs, recall50s, recall75s, precisions, conf_objs, conf_noobjs = [np.array(x) for x in list(zip(*sample_metrics))[:-1]]
            # AP: average precision
            self.logger.experiment.add_scalars("Test/cls_acc/", {"grid_"+str(s): cls_accs.mean()}, self.current_epoch)  
            self.logger.experiment.add_scalars("Test/recall50/", {"grid_"+str(s): recall50s.mean() }, self.current_epoch)  
            self.logger.experiment.add_scalars("Test/recall75/", {"grid_"+str(s):recall75s.mean()}, self.current_epoch)  
            self.logger.experiment.add_scalars("Test/precision/", {"grid_"+str(s): precisions.mean()}, self.current_epoch)  
            self.logger.experiment.add_scalars("Test/conf_obj/", {"grid_"+str(s): conf_objs.mean()}, self.current_epoch)  
            self.logger.experiment.add_scalars("Test/conf_noobj/", {"grid_"+str(s): conf_noobjs.mean()}, self.current_epoch)  


    for idx, v_img in enumerate(vis_list[:4]):
        self.logger.experiment.add_image("Test_%s/orign_img" %(self.current_epoch),
                                            # torch.cat([v_img[:,:,-1:], v_img[:,:,:-1]],dim=2), # BGR2RGB
                                            v_img[:,:,[2,1,0]],
                                            # v_img, # BGR2RGB
                                            idx,
                                            dataformats="HWC")





    