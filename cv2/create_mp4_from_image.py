import cv2
import os
import moviepy.video.io.ImageSequenceClip
import glob

image_folder = 'images'
video_name = 'video.mp4'
fps=1

filenames = [img for img in glob.glob("images/*.png")]

filenames.sort()
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(filenames, fps=fps)
clip.write_videofile(video_name)
