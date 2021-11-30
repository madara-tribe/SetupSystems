import cv2
import numpy as np
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
def opening(img):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
def black2white(img):
    img = np.where(img.any(-1,keepdims=True),img,255)
    return img
    
def main(img_path, mask_path, whitebg=None, smooth=None):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (w, h))
    mask2 = np.where((mask<200),0,1).astype('uint8')
    print(np.unique(mask2))
    nobg = img*mask2[:,:,np.newaxis]
    if smooth:
        nobg = opening(nobg)
    if whitebg:
        nobg = np.where(nobg.any(-1,keepdims=True),nobg,255)
    cv2.imwrite('removedbg.png', nobg)

if __name__=='__main__':
    img_path='input.jpg'
    mask_path='mask.png'
    main(img_path, mask_path, whitebg=True)

