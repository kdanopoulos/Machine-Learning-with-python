from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from SamplePreprocessor import preprocess, preprocess_wrapper
from threading import Thread

class Sample:
	"sample from the dataset"
	def __init__(self, gtText, filePath, image=None):
		self.gtText = gtText
		self.filePath = filePath
		self.image = image;


class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts


class DataLoader:
	"loads data which corresponds to IAM format, see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database" 
	
	def __init__(self, filePath, batchSize, imgSize, maxTextLen, loadToRAM=False):
		"loader for dataset at given location, preprocess images and text according to parameters"

		assert filePath[-1]=='/'

		self.dataAugmentation = False
		self.currIdx = 0
		self.batchSize = batchSize
		self.imgSize = imgSize
		self.samples = []
	
		f=open(filePath+'words.txt')
		chars = set()
		bad_samples = []
		bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
		for line in f:
			# ignore comment line
			if not line or line[0]=='#':
				continue
			
			lineSplit = line.strip().split(' ')
			assert len(lineSplit) >= 9
			
			# filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
			fileNameSplit = lineSplit[0].split('-')
			fileName = filePath + 'words/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' + lineSplit[0] + '.png'

			# GT text are columns starting at 9
			gtText = self.truncateLabel(' '.join(lineSplit[8:]), maxTextLen)
			chars = chars.union(set(list(gtText)))

			# check if image is not empty
			if not os.path.getsize(fileName):
				bad_samples.append(lineSplit[0] + '.png')
				continue

			# put sample into list
			if loadToRAM:
				img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
				self.samples.append(Sample(gtText, fileName, img))
			else:
				self.samples.append(Sample(gtText, fileName))

		# some images in the IAM dataset are known to be damaged, don't show warning for them
		if set(bad_samples) != set(bad_samples_reference):
			print("Warning, damaged images found:", bad_samples)
			print("Damaged images expected:", bad_samples_reference)

		# split into training and validation set: 95% - 5%
		splitIdx = int(0.95 * len(self.samples))
		self.trainSamples = self.samples[:splitIdx]
		self.validationSamples = self.samples[splitIdx:]

		# put words into lists
		self.trainWords = [x.gtText for x in self.trainSamples]
		self.validationWords = [x.gtText for x in self.validationSamples]
		self.words = self.trainWords + self.validationWords

		# number of randomly chosen samples per epoch for training 
		self.numTrainSamplesPerEpoch = 25000 
		
		# start with train set
		self.trainSet()

		# list of all chars in dataset
		self.charList = sorted(list(chars))
		
		# declare getNext() at runtime for best performance
		self.getNext = self.create_getNext(loadToRAM)


	def truncateLabel(self, text, maxTextLen):
		# ctc_loss can't compute loss if it cannot find a mapping between text label and input 
		# labels. Repeat letters cost double because of the blank symbol needing to be inserted.
		# If a too-long label is provided, ctc_loss returns an infinite gradient
		cost = 0
		for i in range(len(text)):
			if i != 0 and text[i] == text[i-1]:
				cost += 2
			else:
				cost += 1
			if cost > maxTextLen:
				return text[:i]
		return text


	def trainSet(self):
		"switch to randomly chosen subset of training set"
		self.dataAugmentation = True
		self.currIdx = 0
		random.shuffle(self.trainSamples)
		self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]

	
	def validationSet(self):
		"switch to validation set"
		self.dataAugmentation = False
		self.currIdx = 0
		self.samples = self.validationSamples


	def getIteratorInfo(self):
		"current batch index and overall number of batches"
		return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


	def hasNext(self):
		"iterator"
		return self.currIdx + self.batchSize <= len(self.samples)
		
	
	def create_getNext(self, loadToRAM):
		if loadToRAM:
			def _getNext():
				"iterator"
				batchRange = range(self.currIdx, self.currIdx + self.batchSize)
				gtTexts = [self.samples[i].gtText for i in batchRange]
				imgs = [preprocess(self.samples[i].image, self.imgSize, self.dataAugmentation) for i in batchRange]
				self.currIdx += self.batchSize
				return Batch(gtTexts, imgs)
		else:
			def _getNext():
				"iterator"
				batchRange = range(self.currIdx, (self.currIdx + self.batchSize))
				gtTexts = [self.samples[i].gtText for i in batchRange]
				threadRange = range(0, 2)
				res = [[] for _ in threadRange]
				threads = [Thread(target=preprocess_wrapper, args=(res, self.samples, self.imgSize, self.dataAugmentation, i, self.currIdx, self.batchSize/2)) for i in threadRange]
				
				for thread in threads:
					thread.start()
				
				for thread in threads:
					thread.join()
				
				imgs = [img for thread_res in res for img in thread_res]
				self.currIdx += self.batchSize
				return Batch(gtTexts, imgs)
		return _getNext


