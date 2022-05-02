import sys, cv2
import time 
import numpy as np

class LockGuard():
    def __init__(self, timeout):
        self.TIMEOUT = timeout
        self.oldtime = time.time()
        self.size = 256
    def display_fps(self, frame, curr_fps, acutual_fps, scrap_frame, x1=0, y1=0, x2=120, y2=20):
        background_color = (0, 0, 0)
        font_color = (255, 255, 255)
        font_size = 0.4
        font_width = 1
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), background_color, -1)
        texts = "FPS: " + str(curr_fps)+'/'+str(acutual_fps)
        frame = cv2.putText(frame, texts, (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, font_width)

        frame = cv2.rectangle(frame, (x1, self.size), (x2-40, self.size-15), background_color, -1)
        frame = cv2.putText(frame, "Scrap:"+str(scrap_frame), (x1 + 5, self.size - 5), cv2.FONT_HERSHEY_COMPLEX, font_size, font_color, font_width)
        return frame

    def lockguard(self, frame, acutual_fps, scrap_frame):
        if (time.time() - self.oldtime) > self.TIMEOUT:
            frame = cv2.resize(frame, (self.size, self.size))
            fps = np.round((time.time() - self.oldtime) / self.TIMEOUT, decimals=2)
            frame = self.display_fps(frame, fps, acutual_fps, scrap_frame)
            self.oldtime = time.time()
            return frame

def main():
    lock = LockGuard(timeout=0.25)
    cap = cv2.VideoCapture('horse_zebra_play2.mp4')
    L = []
    while True:
        ret, frame = cap.read()
        acutual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame = lock.lockguard(frame, np.round(acutual_fps, decimals=2), len(L))
        if frame is None:
            L.append(frame)
        else:
            cv2.imshow('frame', frame)
            L = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow('frame')

if __name__=='__main__':
    main()
