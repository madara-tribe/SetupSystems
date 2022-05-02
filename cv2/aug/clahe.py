import cv2

def clahe(bgr):
    #plt.imshow(bgr),plt.show()
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    #plt.imshow(lab),plt.show()
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
