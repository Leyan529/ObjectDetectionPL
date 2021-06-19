# import os
# from argparse import ArgumentParser

from unicodedata import name

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, GPUStatsMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os

from model.RetinaNet import RetinaNet
from model.SSD import SSD
from model.YOLOV2 import YOLOv2
from model.YOLOV3 import YOLOv3
from model.YOLOV4 import YOLOv4
# from model.YOLOV5 import YOLOV5

from dataset.AsiaTraffic import AsiaModule
from dataset.Pascal import VOCModule
from dataset.WiderPerson import WiderPersonModule
from dataset.Coco import COCOModule
from dataset.Container import MosquitoModule
from dataset.BDD100K import BDD100KModule

if __name__ == '__main__':
    # model = RetinaNet
    # model = SSD 
    model = YOLOv4
    # model = YOLOv3
    # model = YOLOv2

    # dm = AsiaModule(bsz=2, img_size=model.img_size)
    # dm = VOCModule(bsz=2, img_size=model.img_size)
    # dm = COCOModule(bsz=2, img_size=model.img_size)
    # dm = MosquitoModule(bsz=2, img_size=model.img_size)
    # dm = WiderPersonModule(bsz=2, img_size=model.img_size)
    dm = BDD100KModule(bsz=2, img_size=model.img_size)
    dm.setup('fit')  

    
    model = model(dm.get_class(), dm.name)
    # # model = SSD(dm.get_classes(), dm.name)
    # # model = YOLOV2(dm.get_classes(), dm.name)
    # # model = YOLOV3(dm.get_classes(), dm.name)
    # # model = YOLOV4(dm.get_classes(), dm.name)
    # # model = YOLOV5(dm.get_classes(), dm.name)

    setattr(model, "learning_rate", 1e-3)
    model.read_Best_model_path()

    root_dir = os.path.join('log_dir',dm.name)
    logger = TensorBoardLogger(root_dir, name= model.checkname, default_hp_metric =False )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        dirpath= os.path.join(root_dir, model.checkname), 
        filename= model.checkname + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        verbose=True
    )
    setattr(model, "checkpoint_callback", checkpoint_callback)
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    gpu_stats = GPUStatsMonitor() 

    trainer = Trainer(max_epochs = 10, gpus=-1, auto_select_gpus=True,
                    logger=logger, num_sanity_val_steps=0, 
                    # weights_summary='full', 
                    auto_scale_batch_size = 'power', # only open when find batch_num
                    auto_lr_find=True,
                    accumulate_grad_batches=8,  # random error with (could not broadcast input array from shape)
                    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, gpu_stats],
                    limit_train_batches=5,
                    limit_val_batches=5,
                    limit_test_batches = 5
                    )

    # trainer.tune(model, datamodule=dm)                    
    trainer.fit(model, datamodule=dm)

    dm.setup('test') 
    trainer.test(model, datamodule=dm)

    # tensorboard --logdir=D:\WorkSpace\JupyterWorkSpace\ObjectDetectionPL\log_dir  --bind_all
