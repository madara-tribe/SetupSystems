import numpy as np
import cv2

def sharp(img_batch):
    # シャープの度合い
    k = 0.3
    # 粉雪（シャープ化）
    shape_operator = np.array([[0, -k, 0],[-k, 1 + 4 * k, -k],[0,-k, 0]])

    img_sharp=[]
        # 作成したオペレータを基にシャープ化
    for i in img_batch:
        img_sharp.append(cv2.convertScaleAbs(cv2.filter2D(i, -1, shape_operator)))
    return img_sharp

