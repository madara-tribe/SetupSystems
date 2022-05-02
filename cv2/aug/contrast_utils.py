import cv2 
import numpy as np

def contrast(img, gamma = 0.5):
    gamma_cvt = np.zeros((256,1),dtype = 'uint8')
 
    for i in range(256):
        gamma_cvt[i][0] = 255 * (float(i)/255) ** (1.0/gamma)
    return cv2.LUT(img, gamma_cvt)


def average_contrast(img_batch):
    average_square = (2,2)
    averaged_img=[]
    for i in img_batch:
        averaged_img.append(cv2.blur(i, average_square))
    return averaged_img


def change_contrast(img_batch):
    min_table = 10
    max_table = 205
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    high_cont_img = []
    for i in img_batch:
        high_cont_img.append(cv2.LUT(i, LUT_HC))
    low_cont_img = []
    for i in img_batch:
        low_cont_img.append(cv2.LUT(i, LUT_LC))

    return high_cont_img, low_cont_img



