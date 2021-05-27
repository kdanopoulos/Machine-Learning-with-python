from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2

def preprocess(img, imgSize, dataAugmentation=False):
	"put img into target img of size imgSize, transpose for TF and normalize gray-values"

	# there are damaged files in IAM dataset - just use black image instead
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	# increase dataset size by applying random stretches to the images
	if dataAugmentation:
		h_stretch = (random.random() - 0.5) # -0.5 .. +0.5
		w_stretch = (random.random() - 0.5) # -0.5 .. +0.5
		wStretched = max(int(img.shape[1] * (1 + h_stretch)), 1) # random width, but at least 1
		hStretched = max(int(img.shape[0] * (1 + w_stretch)), 1) # random height, but at least 1
		img = cv2.resize(img, (wStretched, hStretched)) # stretch horizontally by factor 0.5 .. 1.5
	
	# create target image and copy sample image into it
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = cv2.resize(img, newSize)
	
	# insert random translation
	translate = [int(random.random()*(targetS - newS)) for targetS, newS in zip(imgSize, newSize)]
	
	# generate noisy background
	target = np.random.randint(220, 255, [ht, wt])
	
	# create target image
	target[translate[1]:translate[1]+newSize[1], translate[0]:translate[0]+newSize[0]] = img
	
	# transpose for TF
	img = cv2.transpose(target)
	
	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img
	return img

def preprocess_wrapper(res, samples, imgSize, augment, thread_id, idx, num):
	for i in range(int(idx+thread_id*num), int(idx+thread_id*num+num)):
		img = samples[i].image if samples[i].image != None else cv2.imread(samples[i].filePath, cv2.IMREAD_GRAYSCALE)
		args = (img, imgSize, augment)
		res[thread_id].append(preprocess(*args))