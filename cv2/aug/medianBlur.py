import cv2 
def noise(img):
　　　ksize = 3
　　　for j in range(3):
	img[:,:,j]=cv2.medianBlur(img[:, :, j], ksize)
　　　return img
