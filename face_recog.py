import cv2
import json
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_xml_location = os.path.join('.', 'evn', 'lib', 'python3.9', 'site-packages', 'cv2', 'data','haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(face_xml_location)
font = cv2.FONT_HERSHEY_SIMPLEX


# 通过人脸编号获取到人脸姓名
def get_name(filepath, id):
    # read from file
    with open(filepath) as file:
        info_list = json.load(file)
    return info_list[id]


def camera_face_recognize():
    camera = cv2.VideoCapture(0)

    color = (85, 185, 190)[::-1]

    while True:
        ret, im = camera.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            img_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), color, 5)
                cv2.rectangle(im, (x, y + h - 30), (x + 170, y + h + 10), color, -1)
                cv2.putText(im, str(img_id) + '||' + str(round(conf, 2)), (x, y + h), font, 1, (21, 38, 56)[::-1], 3)
        cv2.imshow('im', im)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def photo_face_recognize(path, infopath):
    for image_path in os.listdir(path):
        # 忽略掉非.jpg文件
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue

        img = cv2.imread(os.path.join(path, image_path))
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.2, 5)
        for (x, y, w, h) in faces:
            img_id, conf = recognizer.predict(grey[y:y + h, x:x + w])
            str_out1 = 'name: ' + get_name(infopath, img_id)
            # str_out1 = 'name: JA'
            str_out2 = 'with conf: ' + str(round(conf, 2))
            cv2.putText(img, str_out1, (10, y + h), font, 1, (255, 255, 255)[::-1], 3)
            cv2.putText(img, str_out2, (10, y + h + 25), font, 1, (255, 255, 255)[::-1], 3)
            pic_save_location = os.path.join('.', 'result', image_path)
            cv2.imwrite(pic_save_location, img)


# 人脸识别函数
# face_recog_path：需要识别的图片文件文件夹路径
# module_path：训练的识别模型文件夹路径
def face_recog(face_recog_path='./recog_pic', module_path='./train_data'):
    train_data_location = os.path.join(module_path, 'train.yml')
    print(train_data_location)
    recognizer.read(train_data_location)
    photo_face_recognize(face_recog_path, os.path.join(module_path, 'data.json'))
