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
from model.YOLOV5 import YOLOv5

from dataset.AsiaTraffic import AsiaModule
from dataset.Pascal import VOCModule
from dataset.WiderPerson import WiderPersonModule
from dataset.Coco import COCOModule
from dataset.Container import MosquitoModule
from dataset.BDD100K import BDD100KModule

import yaml
import argparse
import torch

def load_config(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in config.items():
            if isinstance(value, dict):
                for inside_key, inside_key_value in value.items():
                    setattr(args, inside_key, inside_key_value)
            else:
                setattr(args, key, value)
    if args.model_name == "RetinaNet": args.img_size = 600
    elif args.model_name == "SSD": args.img_size = 300
    elif args.model_name == "YOLOv5": args.img_size = 640
    else: args.img_size = 416
    return args

def load_data(args):
    dm = None
    if args.data_module == "Asia": dm = AsiaModule(batch_size= args.batch_size, img_size=args.img_size)
    elif args.data_module == "VOC": dm = VOCModule(batch_size= args.batch_size, img_size=args.img_size)
    elif args.data_module == "COCO": dm = COCOModule(batch_size= args.batch_size, img_size=args.img_size)
    elif args.data_module == "Mosquito": dm = MosquitoModule(batch_size= args.batch_size, img_size=args.img_size)
    elif args.data_module == "WiderPerson": dm = WiderPersonModule(batch_size= args.batch_size, img_size=args.img_size)
    elif args.data_module == "BDD100K": dm = BDD100KModule(batch_size= args.batch_size, img_size=args.img_size)
    dm.setup(args.stage)
    return dm

def load_model(args, dm):
    model = None
    if args.model_name == "RetinaNet": model = RetinaNet(dm.get_class(), args)
    elif args.model_name == "SSD": model = SSD(dm.get_class(), args)
    elif args.model_name == "YOLOv2": model = YOLOv2(dm.get_class(), args)
    elif args.model_name == "YOLOv3": model = YOLOv3(dm.get_class(), args)
    elif args.model_name == "YOLOv4": model = YOLOv4(dm.get_class(), args)
    elif args.model_name == "YOLOv5": model = YOLOv5(dm.get_class(), args)

    return model

# if __name__ == '__mainq__':
#     # anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
#     net = YOLOv5(80, None).cuda()
#     net.training = False
#     sample = torch.rand((1, 3, 416, 416)).cuda()
#     output = net(sample)    

if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/config.yaml',
            help='YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args)
    dm = load_data(args)
    model = load_model(args, dm)
    
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

    trainer = Trainer.from_argparse_args(config, 
                                        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, gpu_stats],
                                        logger=logger)

    if args.tune:
        trainer.tune(model, datamodule=dm)                    
    trainer.fit(model, datamodule=dm)

    dm.setup('test') 
    trainer.test(model, datamodule=dm)

    # tensorboard --logdir=D:\WorkSpace\JupyterWorkSpace\ObjectDetectionPL\log_dir  --bind_all
