import cv2
import json
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
train_data_location = os.path.join('.', 'train_data', 'train.yml')
recognizer.read(train_data_location)
face_xml_location = os.path.join('.', 'evn', 'lib', 'python3.9', 'site-packages', 'cv2', 'data','haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(face_xml_location)
font = cv2.FONT_HERSHEY_SIMPLEX


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


def photo_face_recognize(path):
    # read from file
    with open("data.json") as file:
        info_list = json.load(file)

    for image_path in os.listdir(path):
        if os.path.split(image_path)[-1].split(".")[-1] != 'JPG':
            continue

        img = cv2.imread(os.path.join(path, image_path))
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.2, 5)
        for (x, y, w, h) in faces:
            img_id, conf = recognizer.predict(grey[y:y + h, x:x + w])
            str_out1 = 'name: ' + info_list[img_id][0] + ' age: ' + info_list[img_id][1]
            str_out2 = 'with conf: ' + str(round(conf, 2))
            cv2.putText(img, str_out1, (10, y + h), font, 1, (21, 38, 56)[::-1], 3)
            cv2.putText(img, str_out2, (10, y + h + 25), font, 1, (21, 38, 56)[::-1], 3)
            pic_save_location = os.path.join('.', 'result', image_path)
            cv2.imwrite(pic_save_location, img)


if __name__ == '__main__':
    photo_face_recognize('./recog_pic')
    exit(0)