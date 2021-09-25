# Face_rec

> 通过Open-CV进行简单人脸识别

## face_preprocessing文件

第一个写的文件，有各代码的详细注释

### 功能

检测人脸，并把人脸部分裁剪存储起来

### Input文件

- dir: ‘pic.JPG’

### Output文件

- dir: “test.jpg”

## create_imgs

### 功能

通过设备相机，拍摄50张人脸灰度图片，用于训练

### Output文件

- dir: “./face_pic”

## training model

### 功能

通过输入的图片进行训练，文件名为标签

采用的训练是OpenCV自带的LBPHFaceRecognizer 局部二值模式

### Input文件

- dir: “./training_pic” 
  - 内为训练dataset

### Output文件

- dir: “./train_data/train.yml” 
  - 内为训练模型权重
- dir: “data.json”
  - 内为图片对应标签内容

## face_recog

### 功能

通过训练模型猜测测试图标签是谁

### Input文件

- dir: “./recog_pic” 
  - 内为需要识别的人脸

### Output文件

- dir: “./result 
  - 内为识别的结果

