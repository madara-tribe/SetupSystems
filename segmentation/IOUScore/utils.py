import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

CLASS_NAMES = ("None",
               "Road",
               "Sign",
               "Car",
               "Pedestrian")
               
colorB = [0, 0, 255, 255, 69]
colorG = [0, 0, 255, 0, 47]
colorR = [0, 255, 0, 0, 142]
N_CLASSES = len(colorB)

CLASS_COLOR = list()
for i in range(0, N_CLASSES):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")


def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros((seg.shape[0],seg.shape[1],3)).astype('float')
    #colors = sns.color_palette("hls", n_classes) #DB
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

    return seg_img
    
def NormalizeImageArr(path, H, W):
    NORM_FACTOR = 255
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    img = img/NORM_FACTOR
    return img

def LoadSegmentationArr(path , nClasses,  width ,height, train_is=True):
    seg_labels = np.zeros((height, width, nClasses))
    if train_is:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        img = img[:, : , 0]
        for c in range(nClasses):
            seg_labels[: , : , c ] = (img == c ).astype(int)
        return seg_labels
    else:
        img = cv2.imread(path, 0)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        return img
        

def IoU(Yi,y_predi):
    ## mean Intersection over Union
    ## Mean IoU = TP/(FN + TP + FP)
    IoUs = []
    Nclass = int(np.max(Yi)) + 1
    print(Nclass)
    for c in range(Nclass):
        TP = np.sum( (Yi == c)&(y_predi == c))
        FP = np.sum( (Yi != c)&(y_predi == c))
        FN = np.sum( (Yi == c)&(y_predi != c))
        IoU = TP/float(TP + FP + FN)
        #print("class {:02.0f}: #TP={:7.0f}, #FP={:7.0f}, #FN={:7.0f}, IoU={:4.3f}".format(c,TP,FP,FN,IoU))
        print("class (%2d) %12.12s: #TP=%7.0f, #FP=%7.0f, #FN=%7.0f, IoU=%4.3f" % (c, CLASS_NAMES[c],TP,FP,FN,IoU))
        IoUs.append(IoU)
    mIoU = np.mean(IoUs)
    print("_________________")
    print("Mean IoU: {:4.3f}".format(mIoU))
    return


# Visualize the model performance
def visualize_model_performance(X_test, y_pred1_i, y_test1_i, n_classes):

    for k in range(len(X_test)):

        i = k
        img_is  = (X_test[i] + 1)*(255.0/2)
        seg = y_pred1_i[i]
        segtest = y_test1_i[i]

        fig = plt.figure(figsize=(10,30))
        ax = fig.add_subplot(1,3,1)
        ax.imshow(img_is/255.0)
        ax.set_title("original")

        ax = fig.add_subplot(1,3,2)
        ax.imshow(give_color_to_seg_img(seg,n_classes))
        ax.set_title("predicted class")

        ax = fig.add_subplot(1,3,3)
        ax.imshow(give_color_to_seg_img(segtest,n_classes))
        ax.set_title("true class")

        plt.savefig("output/output_" + str(i) + ".png")

    plt.show()


