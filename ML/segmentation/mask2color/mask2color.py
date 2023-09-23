import cv2
import numpy as np

HEIGHT = int(400)
WIDTH  = int(640)
N_CLASSES = 5


colorB = [0, 0, 255, 255, 69]
colorG = [0, 0, 255, 0, 47]
colorR = [0, 255, 0, 0, 142]

CLASS_COLOR = list()
for i in range(0, N_CLASSES):
    CLASS_COLOR.append([colorR[i], colorG[i], colorB[i]])
COLORS = np.array(CLASS_COLOR, dtype="float32")

def give_color_to_seg_img(seg, n_classes):
    if len(seg.shape)==3:
        seg = seg[:,:,0]
    seg_img = np.zeros((seg.shape[0],seg.shape[1],3)).astype('float')
    colors = COLORS #DB
    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:,:,0] += (segc*( colors[c][0]/255.0 ))
        seg_img[:,:,1] += (segc*( colors[c][1]/255.0 ))
        seg_img[:,:,2] += (segc*( colors[c][2]/255.0 ))

    return seg_img

def LoadSegmentationArr(path, width ,height):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img

if __name__=='__main__':
    path = 'anno.png'
    mask = LoadSegmentationArr(path, WIDTH, HEIGHT)
    print(mask.shape, mask.min(), mask.max())
    colormap = give_color_to_seg_img(mask, N_CLASSES)
    colormap = colormap*255
    print(colormap.shape, colormap.min(), colormap.max())
    cv2.imwrite('colormap.png', colormap.astype(np.uint8))
