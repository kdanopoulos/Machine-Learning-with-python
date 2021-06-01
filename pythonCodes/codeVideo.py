

import cv2 as cv
import numpy as np
import math



count = 0

video = "video/test_video.mp4"

vCap = cv.VideoCapture(video)

frameRate = vCap.get(5)
print('The frame rate is : {0}'.format(frameRate))

while(vCap.isOpened()==True):
	frameId = vCap.get(1)
	ret , frame = vCap.read()
	if(ret == False):
		break


	if(frameId % math.floor(frameRate) == 0):
		fileName = "video/frame%d.jpg" %count;
		count+=1
		cv.imwrite(fileName,frame)

vCap.release()	
