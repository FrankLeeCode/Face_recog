'''
在图片上识别出人脸，并保存脸部分裁剪后得到的新图片
'''

import cv2
import os

# os.path.join函数会根据系统返回一个路径字符串
filename = os.path.join('.', 'pic.jpg')
savefilename = os.path.join('.', 'test.jpg')
IMAGE_SIZE = 64


# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)

    # 获取图像尺寸
    h, w, _ = image.shape
    print(image.shape)
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass

    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，使图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))


def detect(filename):
    # 找到人脸识别文件路径./evn/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_alt2.xml
    face_xml_location = os.path.join('.', 'evn', 'lib', 'python3.9', 'site-packages', 'cv2', 'data', 'haarcascade_frontalface_alt2.xml')

    # 声明cascadeclassifie对象face_cascade，用于检测人脸
    face_cascade = cv2.CascadeClassifier(face_xml_location)

    img = cv2.imread(filename)
    # openCV识别灰度图片，也就是黑白图片，所以要进行转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 人脸检测，（灰度图像，scaleFactor尺度参数为1.3，minNeighbors参数为5【扫描到5次都得到的相同矩形才认为是真人脸】
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 在人脸周围绘制矩形
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 0)
        cv2.namedWindow('facedetected')
        cv2.imshow('facedetected', img)
        cv2.imwrite(savefilename, img[y:y+h, x:x+w])
        c = cv2.waitKey(0)


# 图片人脸检测程序，识别图片pic.jpg
detect(filename)
