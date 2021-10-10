import cv2
import os
import numpy as np
import json
from PIL import Image

face_xml_location = os.path.join('.', 'evn', 'lib', 'python3.9', 'site-packages', 'cv2', 'data','haarcascade_frontalface_alt2.xml')
detector = cv2.CascadeClassifier(face_xml_location)
recognizer = cv2.face.LBPHFaceRecognizer_create()


# 更改其他类型图像文件成jpg
def imags_preprocess(file_path):
    for image_path in os.listdir(file_path):
        file_split = os.path.split(image_path)[-1].split(".")
        if file_split[-1] != 'jpg':
            if file_split[-1] == 'JPG' or file_split[-1] == 'png':
                os.rename(os.path.join(file_path, image_path), os.path.join(file_path, file_split[0] + '.jpg'))


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    info_list = []
    id = -1
    face_name = ''
    for image_path in image_paths:
        # 忽略掉非.jpg文件
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue

        # 将文件夹图片挨着转换成灰度图像【L = R * 299/1000 + G * 587/1000+ B * 114/1000】
        image = Image.open(image_path).convert('L')
        # 将图片转换成8位无符号【每个数字代表灰度像素数值】二维数组
        image_np = np.array(image, 'uint8')
        # 检测人脸
        faces = detector.detectMultiScale(image_np)
        # 取图片名
        image_name = os.path.split(image_path)[-1].split('.')[0]

        # 给图片分配id标签
        face_name = image_name.split('_')[0]
        if not (face_name in info_list):
            face_name = image_name.split('_')[0]
            id = id + 1
            # 记录这个人的姓名和id号
            info_list.append(face_name)

        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(info_list.index(face_name))

    return face_samples, ids, info_list


# 训练函数
# face_imag_path：训练的人脸图片文件夹位置
# save_result_path： 训练结果模型路径
def train(face_imag_path, save_result_path):
    train_data_location = save_result_path
    face_data_location = face_imag_path
    imags_preprocess(face_data_location)
    faces, Ids, infos = get_images_and_labels(face_data_location)
    recognizer.train(faces, np.array(Ids))
    recognizer.save(train_data_location)

    # write to file
    os.chdir('./train_data')
    with open("data.json", "w") as file:
        json.dump(infos, file)
