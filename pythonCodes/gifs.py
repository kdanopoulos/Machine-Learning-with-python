#!/usr/bin/env python

from PIL import Image
import math
import numpy as np
import keras as kr

def createFramesFromGifs(name):
	fileName = 'video/gifs/'+name+'/text.txt'
	imagesPerGif = 10

	with open(fileName,'r') as file:
		allLines = file.readlines()
		numberOfLines = len(allLines)
		text = ''
		for i in range(numberOfLines):
			currentLine = allLines[i]
			currentLine = currentLine[:-1]
			image = Image.open("video/gifs/"+name+"/"+currentLine)
			ix = int(image.n_frames/imagesPerGif)
			for j in range(imagesPerGif):
				image.seek(j*ix)
				(wid,hei) = image.size
				if(wid != 480 or hei != 240):
					image2 = image.resize((480,240))
					image2.save("video/gifs/"+name+"/images/img%d_"%(i+1)+"%d.png"%j)
				else:
					image.save("video/gifs/"+name+"/images/img%d_"%(i+1)+"%d.png"%j)
				text+='img%d_'%(i+1)+'%d.png\n'%j
		f = open('video/gifs/'+name+'/images/text.txt','w+')
		f.write(text)
		f.close()


def calulateMeanAndStr():
	celebrityfl = 'video/gifs/cele.txt'
	print("start calculating mean variables of every channel")
	with open(celebrityfl,'r') as file:
		allL = file.readlines()
		number = len(allL)
		meanR = 0
		meanG = 0
		meanB = 0
		strR = 0
		strG = 0
		strB = 0
		# number = number of celebrities
		for i in range(number):
			curL = allL[i]
			if(i!=number-1):
				curL = curL[:-1]
			#createFramesFromGifs(curL)
			print('current celebrity folder is %s'%curL)
			fL = 'video/gifs/'+curL+'/images/text.txt'
			with open(fL,'r') as file2:
				allL2 = file2.readlines()
				number2 = len(allL2)
				# number2 = number of images of this particular celebrity
				for z in range(number2):
					curL2 = allL2[z]
					curL2 = curL2[:-1]
					img = Image.open('video/gifs/'+curL+'/images/'+curL2)
					img2 = img.convert('RGB')
					(x,y) = img.size
					# run for every pixel in this particular image
					for m in range(x):
						for n in range(y):
							r,g,b = img2.getpixel((m,n))
							meanR += r
							meanG += g
							meanB += b
		numerOfTotalPixels = number * number2 *  (x*y)#(number of celebrities) * (number of images per celebrity) * (number of pixels in one image)
		meanR = meanR / numerOfTotalPixels
		meanG = meanG / numerOfTotalPixels
		meanB = meanB / numerOfTotalPixels
		print('mean variables calulated now we will find the standare deviation')
		for i in range(number):
			curL = allL[i]
			if(i!=number-1):
				curL = curL[:-1]
			fL = 'video/gifs/'+curL+'/images/text.txt'
			print('current celebrity folder is %s'%curL)
			with open(fL,'r') as file2:
				allL2 = file2.readlines()
				number2 = len(allL2)
				for z in range(number2):
					curL2 = allL2[z]
					curL2 = curL2[:-1]
					img = Image.open('video/gifs/'+curL+'/images/'+curL2)
					img2 = img.convert('RGB')
					(x,y) = img.size
					for m in range(x):
						for n in range(y):
							r,g,b = img2.getpixel((m,n))
							strR += (r-meanR)**2
							strG += (g-meanG)**2
							strB += (b-meanB)**2
		strR = math.sqrt(strR/(numerOfTotalPixels-1))
		strG = math.sqrt(strG/(numerOfTotalPixels-1))
		strB = math.sqrt(strB/(numerOfTotalPixels-1))
		print('the red mean is %f'%meanR)
		print('the green mean is %f'%meanG)
		print('the blue mean is %f'%meanB)
		print('the red str is %f'%strR)
		print('the green str is %f'%strG)
		print('the blue str is %f'%strB)
		f = open('storage.txt','w+')
		tempory = '%f,'%meanR+'%f,'%meanG+'%f,'%meanB+'%f,'%strR+'%f,'%strG+'%f'%strB
		f.write(tempory)
		f.close()
		print("The variables were saved at the folder : storage.txt")	



def normalizeAll():
	celebrityfl = 'video/gifs/cele.txt'
	print('we will normalize every pixel of every image')
	with open(celebrityfl,'r') as file:
		allL = file.readlines()
		number = len(allL)
		text = ''
		f = open('storage.txt','r')
		rl = f.readlines()
		text = rl[0].split(',')
		text[len(text)-1] = text[len(text)-1][:-1]
		f.close()
		meanR = float(text[0])
		meanG = float(text[1])
		meanB = float(text[2])
		strR = float(text[3])
		strG = float(text[4])
		strB = float(text[5])

		for i in range(number):
			curL = allL[i]
			if(i!=number-1):
				curL = curL[:-1]
			fL = 'video/gifs/'+curL+'/images/text.txt'
			print('The current folder is : '+curL)
			with open(fL,'r') as file2:
				allL2 = file2.readlines()
				number2 = len(allL2)
				for z in range(number2):
					curL2 = allL2[z]
					curL2 = curL2[:-1]
					img = Image.open('video/gifs/'+curL+'/images/'+curL2)
					img2 = img.convert('RGB')
					(x,y) = img.size
					for m in range(x):
						for n in range(y):
							r,g,b = img2.getpixel((m,n))
							r = (r-meanR)/strR
							g = (g-meanG)/strG
							b = (b-meanB)/strB
							img2.putpixel((m,n),(int(r),int(g),int(b)))
					img2.save(img.filename)
					print('image %d normalized!'%z)			
				

			

celeb = 4
imagePerCel = 400



data = []
dataY = np.zeros(celeb*imagePerCel)
couter=0
first = True


celebrityfl = 'video/gifs/cele.txt'
with open(celebrityfl,'r') as file:
	allL = file.readlines()
	number = len(allL)


	for i in range(number):
		curL = allL[i]
		if(i!=number-1):
			curL = curL[:-1]
		fL = 'video/gifs/'+curL+'/images/text.txt'
		print('The current folder is : '+curL)
		with open(fL,'r') as file2:
			allL2 = file2.readlines()
			number2 = len(allL2)
			for z in range(number2):
				curL2 = allL2[z]
				curL2 = curL2[:-1]
				img = Image.open('video/gifs/'+curL+'/images/'+curL2)
				img2 = img.convert('RGB')
				img2 = np.asarray(img2)   # convert to numpy array
				img2 = np.expand_dims(img2,axis=0) # add one dimension at axis = 0
				if(first): # is this is the first image
					data = img2
					first = False
				else:
					data = np.append(data,img2,axis=0)
				if(curL=='adam'):
					dataY[couter] = 0
				elif(curL=='aniston'):
					dataY[couter] = 1
				elif(curL=='dicaprio'):
					dataY[couter] = 2
				elif(curL=='joey'):
					dataY[couter] = 3
				couter+=1


#print('the transformation of the dataset has begin...')
#npData = np.array(data)
npData = data
print('the dataset was transformed to numpy array')


# suffle of the 2 arrays
#ran = np.arange(1600,dtype=np.int32)
#print(ran[0])
#ran = np.random.shuffle(ran)
number_images = npData.shape[0]
ran = np.random.permutation(number_images)
temp = npData
tempY = dataY
for i in range(number_images):
	nump = ran[i]
	temp[i][:][:] = npData[nump][:][:]
	tempY[i] = dataY[nump]

npData = temp
dataY = tempY
#---------------------

print("Total\n==========")
print(npData.shape)
print("========================\n")

per  = 90
length = number_images
boundry = int((length*per)/100)
x_train =  npData[0:boundry][:][:]
x_test = npData[boundry+1:][:][:]
y_train = dataY[0:boundry]
y_test = dataY[boundry+1:]

print("Train\n==========")
print(x_train.shape)
print(y_train.shape)

print("\nTest\n==========")
print(x_test.shape)
print(y_test.shape)

model = kr.Sequential()
model.add(kr.layers.Conv2D(32,(3,3),activation='relu',input_shape = (240,480,3) ))
model.add(kr.layers.Conv2D(64,(3,3),activation='relu'))
model.add(kr.layers.MaxPooling2D(pool_size=(2,2)))
model.add(kr.layers.Dropout(0.25))
model.add(kr.layers.Flatten())
model.add(kr.layers.Dense(128,activation='relu'))
model.add(kr.layers.Dropout(0.5))
model.add(kr.layers.Dense(4,activation='softmax'))

model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=5)
(loss,accu) = model.evaluate(x_test,y_test)
print('The accurancy is : ',accu)
print('The loss is : ',loss)



