# ObjectDetection
My Frame work for Image Object Detection with pytorch Lightning + Albumentations
## Overview
I organizize the object detection algorithms proposed in recent years, and focused on **`COCO`, `VOC`, `Mosquito containers`, `WiderPerson` , `Asia Traffic` and `BDD100K`** Dataset.

### Status
- [x] DataSet
- [x] Training/Validiation Step
- [x] TensorBoardLogger callback
- [x] ModelCheckpoint callback
- [x] Learning Rate Monitor
- [x] GPUStats Monitor
- [x] Early stop callback
- [x] Read Best model path
- [x] Lightning with Argument Parser
- [x] Yaml custom config
- [x] Yolo v5 model

## Datasets:

I used 6 different datases: **`COCO`, `VOC`, `Mosquito containers`, `WiderPerson` , `Asia Traffic` and `BDD100K` ** . Statistics of datasets I used for experiments is shown below

- **VOC**:
  Download the voc images and annotations from [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007) or [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012). Make sure to put the files as the following structure:
  
| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| VOC2007                |    20   |      5011/12608       |           4952/-           |
| VOC2012                |    20   |      5717/13609       |           5823/13841       |

  ```
  VOCDevkit
  ├── VOC2007
  │   ├── Annotations  
  │   ├── ImageSets
  │   ├── JPEGImages
  │   └── ...
  └── VOC2012
      ├── Annotations  
      ├── ImageSets
      ├── JPEGImages
      └── ...
  ```
  
- **COCO**:
  Download the coco images and annotations from [coco website](http://cocodataset.org/#download). Make sure to put the files as the following structure:

| Dataset                | Classes | #Train images/objects | #Validation images/objects |
|------------------------|:---------:|:-----------------------:|:----------------------------:|
| COCO2014               |    80   |         83k/-         |            41k/-           |
| COCO2017               |    80   |         118k/-        |             5k/-           |
```
  COCO
  ├── annotations
  │   ├── instances_train2014.json
  │   ├── instances_train2017.json
  │   ├── instances_val2014.json
  │   └── instances_val2017.json
  │── images
  │   ├── train2014
  │   ├── train2017
  │   ├── val2014
  │   └── val2017
  └── anno_pickle
      ├── COCO_train2014.pkl
      ├── COCO_val2014.pkl
      ├── COCO_train2017.pkl
      └── COCO_val2017.pkl
```

- **Mosquito containers**:
This challenge was provided by the Center for Disease Control (CDC) of the Ministry of Health and Welfare of the Republic of China. The following table is a description table of the object category and corresponding ID.

|  ID    |   Category       | 
| ------ | -----------------| 
|   1    | aquarium         | 
|   2    | bottle           | 
|   3    |   bowl           | 
|   4    |   box            | 
|   5    |   bucket         | 
|   6    | plastic_bag      | 
|   7    |   plate          | 
|   8    |   styrofoam      | 
|   9    |   tire           | 
|   10   |   toilet         | 
|   11   |   tub            | 
|   12   | washing_machine  | 
|   13   |   water_tower    | 


  Download the container images and annotations from [MosquitoContainer contest](https://aidea-web.tw/topic/47b8aaa7-f0fc-4fee-af28-e0f077b856ae?focus=intro). Make sure to put the files as the following structure:

```
  MosquitoContainer
  ├── train_cdc
  │   ├── train_annotations
  │   ├── train_images
  │   ├── train_labels  
  │     
  │── test_pub_cdc
  │   ├── test_pub_images
  │   
  └── test_cdc
      ├── test_images
```

- **Asian-Traffic**:
Object detection in the field of computer vision has been extensively studied, and the use of deep learning methods has made great progress in recent years.
In addition, existing open data sets for object detection in ADAS applications usually include pedestrians, vehicles, cyclists, and motorcyclists in Western countries, which is different from Taiwan and other crowded Asian countries (speeding on urban roads).

  Download the container images and annotations from [Asian Countries contest](https://aidea-web.tw/topic/35e0ddb9-d54b-40b7-b445-67d627890454?focus=intro&fbclid=IwAR3oSJ8ESSTjPmf0nyJtggacp0zjEf77E_H4JC_qMtPPx8xrG4ips9qp6tE). Make sure to put the files as the following structure:

```
  Asian-Traffic(Stage1)
  ├── ivslab_train
  │   ├── Annotations
  │   ├── ImageSets
  │   ├── JPEGImages  
  │     
  │── ivslab_test_public
     ├── JPEGImages
```
- **WiderPerson**:
Download the WiderPerson dataset through Google Drive or Baidu Drive (jx5p), unzip the downloaded file, get the following files:
• "./Images": 13,382 images of this dataset.
• "./Annotations": 9,000 annotation text files of training and valiadation subsets.
• "./Evaluation": evaluation codes.
• "./train.txt": file list of training subset.
• "./test.txt": file list of testing subset.
• "./val.txt": file list of validation subset.
• "./ReadMe.txt": file of instruction.![image]

Annotation Format
Each image of training and valiadation subsets in the "./Images" folder (e.g., 000001.jpg) has a corresponding annotation text file in the "./Annotations" folder (e.g., 000001.jpg.txt). The annotation file structure is in the following format:
...
< number of annotations in this image = N >
< anno 1 >
< anno 2 >
...
< anno N >
...
where one object instance per row is [class_label, x1, y1, x2, y2], and the class label definition is:
...
< class_label =1: pedestrians >
< class_label =2: riders >
< class_label =3: partially-visible persons >
< class_label =4: ignore regions >
< class_label =5: crowd >

Download the container images and annotations from [WiderPerson](https://drive.google.com/file/d/1I7OjhaomWqd8Quf7o5suwLloRlY0THbp/view). Make sure to put the files as the following structure:

```
  WiderPerson
    ├── Images
      ├── *.jpg
    ├── Annotations            
      ├── *.jpg.txt               
```


- **BDD100K**:
BDD100K is a diverse driving dataset for heterogeneous multitask learning.
![image](https://user-images.githubusercontent.com/24097516/121802720-6ce0b000-cc70-11eb-85f9-a757bf946796.png)

Format
This is compatible with the labels generated by Scalabel. A label json file is a list of frame objects with the fields below. Please note that this format is a superset of the data fields. For example, box3d may be absent if the label is a 2d bounding box, and intrinsics may not appear if the exact camera calibration is unknown.

- name: string
- url: string
- videoName: string (optional)
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- intrinsics
    - focal: [x, y]
    - center: [x, y]
    - nearClip:
- extrinsics
    - location
    - rotation
- timestamp: int64 (epoch time ms)
- frameIndex: int (optional, frame index in this video)
- labels [ ]:
    - id: int32
    - category: string (classification)
    - manualShape: boolean (whether the shape of the label is created or modified manually)
    - manualAttributes: boolean (whether the attribute of the label is created or modified manually)
    - score: float (the confidence or some other ways of measuring the quality of the label.)
    - attributes:
        - occluded: boolean
        - truncated: boolean
        - trafficLightColor: "red|green|yellow|none"
        - areaType: "direct | alternative" (for driving area)
        - laneDirection: "parallel|vertical" (for lanes)
        - laneStyle: "solid | dashed" (for lanes)
        - laneTypes: (for lanes)
    - box2d:
       - x1: float
       - y1: float
       - x2: float
       - y2: float
   - box3d:
       - alpha: (observation angle if there is a 2D view)
       - orientation: (3D orientation of the bounding box, used for 3D point cloud annotation)
       - location: (3D point, x, y, z, center of the box)
       - dimension: (3D point, height, width, length)
   - poly2d: an array of objects, with the structure
       - vertices: [][]float (list of 2-tuples [x, y])
       - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
       - closed: boolean (closed for polygon and otherwise for path)
Road object categories:

[
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]

Download the container images and annotations from [BDD100K](https://bdd-data.berkeley.edu/portal.html#download). Make sure to put the files as the following structure:

```
  bdd100k
  ├── bdd100k
    ├── images
      ├── track
        ├── train
        ├── val
        ├── test
    ├── labels            
      ├── box_track_20    
        ├── train
        ├── val            
```

## Methods
- RetinaNet
- SSD
- YOLOv2
- YOLOv3
- YOLOv4
- YOLOv5 [Yolov5s, Yolov5m, Yolov5l, Yolov5x]

## Prerequisites
* **Windows 10**
* **CUDA 10.2**
* **NVIDIA GPU 1660 + CuDNN v7.605**
* **python 3.6.9**
* **pytorch 1.81**
* **opencv-python 4.1.1.26**
* **numpy 1.19.5**
* **torchvision 0.9**
* **torchsummary 1.5.1**
* **Pillow 7.2.0**
* **dlib==19.21**
* **tensorflow-gpu 2.2.0**
* **tensorboard 2.5.0** 
* **pytorch-lightning 1.2.6**
* **albumentations 0.5.2**


## Usage
### 0. Prepare the dataset
* **Download custom dataset in the  `data_paths`.** 
* **And create custom dataset `custom_dataset.py` in the `dataset`.**
* **Test dataset mark to video (need to change `custom_dataset` in code)**

### Execute (Train/Val + Test)

#### Asia
```python
python run.py --use AsiaModule --model YOLOV4
```
#### VOC
```python
python run.py --use VOCModule --model YOLOV4
```
#### COCO
```python
python run.py --use COCOModule --model YOLOV4
```
#### Mosquito
```python
python run.py --use MosquitoModule --model YOLOV4
```
#### WiderPerson
```python
python run.py --use WiderPersonModule --model YOLOV4
```
#### BDD100K
```python
python run.py --use BDD100KModule --model YOLOV4
```

## Reference
- PyTorch-YOLOv2 : https://github.com/uvipen/Yolo-v2-pytorch/blob/9589413b5dce0476eb9cccc41945cf30cf131b34/src/yolo_net.py
- PyTorch-YOLOv3 : https://github.com/eavise-kul/lightnet/blob/b54f771a597d23863979d9274eb2ba90c05938e4/lightnet/models/_network_yolo_v3.py
- PyTorch_YOLOv4 : https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/4ccef0ec8fe984e059378813e33b3740929e0c19/models.py
- PyTorch_YOLOv5 : https://github.com/hpc203/my_yolov5-pytorch
- SSD: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/43fd8be9e82b351619a467373d211ee5bf73cef8/model.py#L323
- RetinaNet: https://github.com/kuangliu/pytorch-retinanet/blob/2d7c663350f330a34771a8fa6a4f37a2baa52a1d/retinanet.py#L8
- Albumentations : https://albumentations.ai/docs/examples/example_bboxes/
- pytorch-lightning : https://pytorch-lightning.readthedocs.io/en/latest/
