import torch
import cv2


def test():
    modul = torch.load('./pth/latest_net_G_A.pth')
    im = cv2.imread('../training_pic/001A22.JPG')
    new = modul(im)
    cv2.imshow(new)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('../training_pic/JAyoung.jpg')
    # cv2.namedWindow('a')
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('../training_pic/JAyoungre.jpg', img)