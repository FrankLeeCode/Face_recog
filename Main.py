import face_recog
import pic_trans
import training_model


if __name__ == '__main__':
    result = '/Users/FrankLee/Desktop/University/科研训练-人脸识别/FaceAging-by-cycleGAN/results'
    pic_trans.pic_collect(result)

    training_model.train('training_pic', 'train_data')


    # face_recog.face_recog()