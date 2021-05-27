from __future__ import division
from __future__ import print_function

import sys
import argparse
import cv2
import editdistance
from DataLoader import DataLoader, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
import time


class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'


def train(model, loader):
	"train NN"
	train_startTime = time.time()
	epoch = 0 # number of training epochs since start
	bestCharErrorRate = float('inf') # best valdiation character error rate
	noImprovementSince = 0 # number of epochs no improvement of character error rate occured
	earlyStopping = 5 # stop training after this number of epochs without improvement
	while True:
		epoch += 1
		print('Epoch:', epoch)
		
		# train
		print('Train NN')
		batches_startTime = time.time()
		loader.trainSet()
		while loader.hasNext():
			batch_startTime = time.time()
			iterInfo = loader.getIteratorInfo()
			batch = loader.getNext()
			batch_trainStartTime = time.time()
			loss = model.trainBatch(batch)
			print('Batch:', iterInfo[0],'/', iterInfo[1], 'Loss: %.3f' % loss, '\t batch dur: %.3f' % (time.time() - batch_startTime), ' preprocess dur: %.3f' % (batch_trainStartTime - batch_startTime), ' train dur: %.3f' % (time.time() - batch_trainStartTime))
		
		# validate
		validate_startTime = time.time()
		charErrorRate, wordAccuracy = validate(model, loader)
		
		print('Batches duration :', validate_startTime - batches_startTime)
		print('Validation duration :', time.time() - validate_startTime)
		print('Total epoch duration :', time.time() - batches_startTime)
		
		
		# if best validation accuracy so far, save model parameters
		if charErrorRate < bestCharErrorRate:
			print('Character error rate improved, save model')
			bestCharErrorRate = charErrorRate
			noImprovementSince = 0
			model.save()
			open(FilePaths.fnAccuracy, 'w').write('Validation character error rate of saved model: %f%%\nValidation word accuracy: %f%%\nTraining epochs: %d\nTraining duration: %.3f' % (charErrorRate*100.0, wordAccuracy*100.0, epoch, time.time()-train_startTime))
		else:
			print('Character error rate not improved')
			noImprovementSince += 1

		# stop training if no more improvement in the last x epochs
		if noImprovementSince >= earlyStopping:
			print('No more improvement since %d epochs. Training stopped after %d epochs.' % (earlyStopping, epoch))
			break


def validate(model, loader):
	"validate NN"
	print('Validate NN')
	loader.validationSet()
	numCharErr = 0
	numCharTotal = 0
	numWordOK = 0
	numWordTotal = 0
	while loader.hasNext():
		iterInfo = loader.getIteratorInfo()
		print('Batch:', iterInfo[0],'/', iterInfo[1])
		batch = loader.getNext()
		(recognized, _) = model.inferBatch(batch)
		
		print('Ground truth -> Recognized')	
		for i in range(len(recognized)):
			numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
			numWordTotal += 1
			dist = editdistance.eval(recognized[i], batch.gtTexts[i])
			numCharErr += dist
			numCharTotal += len(batch.gtTexts[i])
			print('[OK]' if dist==0 else '[ERR:%d]' % dist,'"' + batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')
			validationStatistics(iterInfo[0], i, model.batchSize, len(recognized), batch.gtTexts[i], dist)
			
	# print validation result
	charErrorRate = numCharErr / numCharTotal
	wordAccuracy = numWordOK / numWordTotal
	print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate*100.0, wordAccuracy*100.0))
	return charErrorRate, wordAccuracy


def infer(model, fnImg):
	"recognize text in image provided by file path"
	img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
	batch = Batch(None, [img])
	(recognized, probability) = model.inferBatch(batch, True)
	print('Recognized:', '"' + recognized[0] + '"')
	print('Probability:', probability[0])


def main():
	"main function"
    
	# optional command line args
	parser = argparse.ArgumentParser()
	parser.add_argument('--train', help='train the NN', action='store_true')
	parser.add_argument('--validate', help='validate the NN', action='store_true')
	parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
	parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding', action='store_true')
	parser.add_argument('--mlf', help='Use modified loss function during training', action='store_true')
	parser.add_argument('--lm', help='Construct and use syllable language model', action='store_true')
	parser.add_argument('--loadtoram', help='load entire dataset to RAM instead of reading files from disk for each batch', action='store_true')
	parser.add_argument('--logscores', help='Log validation set character scores (CTC input). Only in validation mode.', action='store_true')
	args = parser.parse_args()
	
	decoderType = DecoderType.BestPath
	if args.beamsearch:
		decoderType = DecoderType.BeamSearch
	elif args.wordbeamsearch:
		decoderType = DecoderType.WordBeamSearch
		
	mlf = True if args.mlf else False
	lm = True if args.lm else False
	log = True if args.logscores else False
	

	# train or validate on IAM dataset
	if args.train or args.validate:
		# load training data, create TF model
		loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen, loadToRAM=args.loadtoram)
		
		# save characters of model for inference mode
		open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
		
		# save words contained in dataset into file
		open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

		# execute training or validation
		if args.train:
			model = Model(loader.charList, decoderType, modifiedLoss=mlf, languageModel=lm, wordsList=loader.words)
			train(model, loader)
		elif args.validate:
			model = Model(loader.charList, decoderType, modifiedLoss=mlf, languageModel=lm, log=log, wordsList=loader.words, mustRestore=True)
			validate(model, loader)

	# infer text on test image
	else:
		print(open(FilePaths.fnAccuracy).read())
		model = Model(open(FilePaths.fnCharList).read(), decoderType, modifiedLoss=mlf, languageModel=lm, log=log, wordsList=loader.words, mustRestore=True)
		infer(model, FilePaths.fnInfer)


# Validation statistics of total character errors per edit distance (including edit distance=0, i.e. correct characters)
# for all labels ('errors') and for single-character labels ('singleChars')
# Lists declared as global variables to be directly displayed in the variable explorer (Spyder)
errors = [0]*20
singleChars_errors = [0]*20
def validationStatistics(batchIdx, sampleIdx, batchSize, inferedSize, label, editdistance):
	if inferedSize != batchSize:
		return
	
	errors[editdistance] += 1
	if len(label) == 1:
		singleChars_errors[editdistance] += 1
	

if __name__ == '__main__':
	main()

