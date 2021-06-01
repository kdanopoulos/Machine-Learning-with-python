import math
import numpy as np
import keras as kr
import os
import cv2

class DataGenerator:
	def __init__(self,mainFolder,sampling,framesPerVideo,width,height):
		self.mF = mainFolder
		self.fPV = framesPerVideo
		self.wid = width
		self.hei = height
		self.samp = sampling
		self.first = True
		# videos
		self.xTrain = 0
		self.xTest = 0
		# labels 
		self.yTrain = 0
		self.yTest = 0
		


	def getPathsAndLabels(self,curFolder):
		xPaths = []
		yDict = {}
		allItems = sorted(os.listdir(curFolder)) # read all subfiles ->(Fight,NonFight) inside the current folder ->(train,test)
		labels = kr.utils.np_utils.to_categorical(range(len(curFolder)))
		for i,item in enumerate(allItems):
			folder = os.path.join(curFolder,item) 
			for file in os.listdir(folder): # read every subfile ->(all videos) inside previous folder ->(Fight,NonFight)
				vPath = os.path.join(folder,file)
				xPaths.append(vPath)
				yDict[xPaths] = labels[i]
		return xPaths,yDict


		

	def uniformSamplingAVideo(self,videoPath):
		f = True
		frames = 0
		cap = cv2.VideoCapture(videoPath)
		totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		gap = int(totalFrames/self.framesPerVideo)
		for i in range(self.framesPerVideo):
			cap.set(1,i*gap)
			test,frame = cap.read()
			if(test):
				frame = np.asarray(frame)
				frame = np.expand_dims(frame,axis=0)
				if(f):
					frames = frame
					f = False
				else:
					frames = np.append(frames,frame,axis=0)
			else:
				print("Error reading the frame : %d"%(i*gap))
				break
		cap.release()
		return frames




	def takeAllFramesOfAVideo(self,videoPath):
		f = True
		frames = 0
		cap = cv2.VideoCapture(videoPath)
		totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		for i in range(totalFrames):
			test,frame = cap.read()
			if(test):
				frame = np.asarray(frame)
				frame = np.expand_dims(frame,axis=0)
				if(f):
					frames = frame
					f = False
				else:
					frames = np.append(frames,frame,axis=0)

			else:
				print("Error reading the frame : %d"%(i*gap))
				break
		cap.release()
		return frames





	def createData(self):
		# get train data 
		trainP = os.path.join(self.mainFolder,"/train")
		trainXPath, trainYDict = getPathsAndLabels(trainP)
		for i in range(len(trainXPath)):
			if(self.samp):
				video = uniformSamplingAVideo(trainXPath[i])
			else:
				video = takeAllFramesOfAVideo(trainXPath[i])
			if(i==0): # first run of for 
				totVid = video
			else:
				totVid = np.append(totVid,video,axis=0)

		# get val data 
		valP = os.path.join(self.mainFolder,"/val")
		valXPath, valYDict = getPathsAndLabels(valP)
		for i in range(len(valXPath)):
			if(self.samp):
				videoVal = uniformSamplingAVideo(valXPath[i])
			else:
				videoVal = takeAllFramesOfAVideo(valXPath[i])
			if(i==0): # first run of for 
				totVidVal = videoVal
			else:
				totVidVal = np.append(totVidVal,videoVal,axis=0)










				



