
import tensorflow as tf
import numpy as np
import keras as kr


def getStringUntilFindLetter(string,symbol):
	newString = '';
	for i in range(len(string)):
		if (string[i]==symbol):
			return newString
		newString += string[i]
	return newString


def getIdAndDiagnose(string):
	tAr = string.split(',')
	newString = tAr[:2]
	return newString


def getLineInformation(string):
	tAr = string.split(',')
	newString = tAr[2:]
	lString = newString[len(newString)-1]
	lString = lString[:len(lString)-2]
	newString[len(newString)-1] = lString
	return newString	







currentLine = ''  #'wdbc.data' #'~/Desktop/pythonCodes/wdbc.data'


fileName = 'wdbc.data'
with open(fileName,'r') as file:
	allLines = file.readlines()
	numberOfLines = len(allLines)
	temp2 = getLineInformation(allLines[0])
	numberOfCol = len(temp2)
	x = np.zeros((numberOfLines,numberOfCol),dtype=np.float32)
	y = np.zeros((numberOfLines,1))
	for i in range(numberOfLines):
		currentLine = allLines[i]
		temp1 = getIdAndDiagnose(currentLine)
		temp2 = getLineInformation(currentLine)
		for j in range(numberOfCol):
			x[i][j] = float(temp2[j])
		if(temp1[1]=='M'):
			y[i] = 1
		else:
			y[i] = 0

print("The x size is :")
print(x.shape)
print("The y length is :")
print(y.shape)

per = 10
per = int ((numberOfLines*10)/100)
per = numberOfLines-per

x_train = x[:per][:]
print("\nx_train size is :")
print(x_train.shape)
x_test = x[per+1:][:]
print("x_test size is :")
print(x_test.shape)

y_train = y[:per]
print("y_train length is :")
print(y_train.shape)
y_test = y[per+1:]
print("y_test length is :")
print(y_test.shape)


# Normalization of the feutures of the dataset 

train_mean = np.mean(x_train,axis=0)

train_std = np.std(x_train,axis=0)

x_train = (x_train-train_mean)/train_std


model = kr.Sequential()
model.add(kr.layers.Dense(64,activation='relu',input_shape =(30,)))
model.add(kr.layers.Dense(128,activation='relu',input_shape =(30,)))
model.add(kr.layers.Dense(248,activation='relu',input_shape =(30,)))
model.add(kr.layers.Dropout(0.5))
model.add(kr.layers.Dense(10,activation='softmax'))
model.add(kr.layers.Dropout(0.25))
model.compile(optimizer='Adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=x_train,y=y_train,epochs=1)


print(x_test.shape)
print(y_test.shape)
(loss, accurancy) = model.evaluate(x_test,y_test)
print("The loss is : {0} and the accurancy is : {1}".format(loss,accurancy))










