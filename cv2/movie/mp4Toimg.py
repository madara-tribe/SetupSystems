import cv2
import os, sys
import time

FPS_TIMEOUT=0.1
def save_image_from_movie(video_path, dir_path, basename, ext='jpg'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    while True:
        ret, frame = cap.read()
        cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
        n += 1
        

if __name__=='__main__':
    video_name = str(sys.argv[1])
    image_folder = str(sys.argv[2])
    image_name = str(sys.argv[3])
    save_image_from_movie(video_name, image_folder, image_name, ext='png')
