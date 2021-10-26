import cv2
import os


def pic_collect(src):
    for i in range(1, 5):
        imgdir = os.path.join(src, 'class'+str(i), 'test_latest', 'images')
        for pic in os.listdir(imgdir):
            if pic.split('.')[-1] == 'png':
                if 'fake_B' in pic:
                    img = cv2.imread(os.path.join(imgdir, pic))
                    # cv2.imshow(pic, img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # cv2.waitKey(0)
                    name = pic.split('_')[0] + '_fake' + str(i) + '.jpg'
                    print(name)
                    cv2.imwrite('training_pic/'+name, img)

