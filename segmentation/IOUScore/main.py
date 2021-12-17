import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import NormalizeImageArr, LoadSegmentationArr, IoU, visualize_model_performance
from verify import verify_datasets

HEIGHT = int(400)
WIDTH  = int(640)
N_CLASSES = 5

def load_dataset(dir_train_img, dir_train_seg):
    verify_datasets(dir_train_img, dir_train_seg, N_CLASSES, validate=True, dir_valid_img=dir_train_img, dir_valid_seg=dir_train_seg)

    # load training images
    train_images = os.listdir(dir_train_img)
    train_images.sort()
    train_segmentations  = os.listdir(dir_train_seg)
    train_segmentations.sort()
    X_train, Y_train=[], []
    anno=[]
    for im , seg in tqdm(zip(train_images,train_segmentations)):
        X_train.append(NormalizeImageArr(os.path.join(dir_train_img,im), WIDTH, HEIGHT))
        Y_train.append(LoadSegmentationArr(os.path.join(dir_train_seg,seg) , N_CLASSES, WIDTH, HEIGHT, train_is=True))
        anno.append(LoadSegmentationArr(os.path.join(dir_train_seg,seg) , N_CLASSES, WIDTH, HEIGHT, train_is=False))
    print('train load')
    X_train, Y_train, anno = np.array(X_train), np.array(Y_train), np.array(anno)

    print(X_train.shape,Y_train.shape, anno.shape)
    print(X_train.max(), X_train.min(), Y_train.max(), Y_train.min())
    return X_train, Y_train, anno

if __name__=='__main__':
    dir_train_img='image'
    dir_train_seg='anno'
    X, Y, anno = load_dataset(dir_train_img, dir_train_seg)
    print(X.shape, Y.shape, anno.shape)
    y_pred = np.array([np.argmax(y, axis=2) for y in Y])
    print(y_pred.shape)
    #plt.imshow(y_pred[0]),plt.show()
    IoU(anno, y_pred)
    visualize_model_performance(anno, y_pred, anno, N_CLASSES)
