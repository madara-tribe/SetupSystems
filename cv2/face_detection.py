import cv2

cascade_path = '/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_default.xml'
CASCADE = cv2.CascadeClassifier(cascade_path)

def near_face_crop_resize(image, cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray)
    if len(faces)>0:
        x, y, w, h = faces[0]
        x = image[: y + h, :]
        return x
    return image

def face_crop_imread(path, h=250, w=299, segment=True):
    x = cv2.imread(path)
    x = near_face_crop_resize(x, CASCADE)
    x = skimage_resize(x, h=w, w=w)
    x = x[:h, :]
    return x/255
