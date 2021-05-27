from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf


class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2


class Model: 
	"minimalistic TF model for HTR"

	# model constants
	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32
	ext_probs = []

	def __init__(self, charList, decoderType=DecoderType.BestPath, modifiedLoss=False, languageModel=False, log=False, wordsList=None, mustRestore=False):
		"init model: add CNN, RNN and CTC and initialize TF"
		tf.reset_default_graph()
        
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0
		self.modifiedLossFunction = modifiedLoss
		self.languageModel = languageModel
		self.log = log

		# Whether to use normalization over a batch or a population
		self.is_train = tf.placeholder(tf.bool, name='is_train')

		# input image batch
		self.inputImgs = tf.placeholder(tf.float32, shape=(Model.batchSize, Model.imgSize[0], Model.imgSize[1]))
		
		# construct language model from the list of input words
		self.sess = None
		self.setupLM(wordsList);
		
		# setup CNN, RNN and CTC
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()

		# setup optimizer to train NN
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		# initialize TF
		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self):
		"create CNN layers and return output of these layers"
		cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)
		
		# list of parameters for the layers
		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)
		
		# create layers
		pool = cnnIn4d # input to first CNN layer
		print(pool.get_shape())
		for i in range(numLayers):
			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
			print(pool.get_shape())
		
		self.cnnOut4d = pool


	def setupRNN(self):
		"create RNN layers and return output of these layers"
		print('LSTM')
		cnnOut4d_shape = self.cnnOut4d.get_shape().as_list()
		
		# Squeeze 2D CNN output to 1D (either by removing singleton dimension or by column-wise concatenation)
		if cnnOut4d_shape[2] > 1:
			self.cnnOut4d = tf.transpose(self.cnnOut4d, [0, 1, 3, 2])
			rnnIn3d = tf.reshape(self.cnnOut4d, cnnOut4d_shape[:-2]+[-1])
		else :
			rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])
		print(rnnIn3d.get_shape())
		
		def set_train_true():
			self.bool_is_train = True
			return True
		def set_train_false():
			self.bool_is_train = False
			return False
		tf.cond(tf.equal(self.is_train, tf.constant(True)), set_train_true, set_train_false)
		self.bool_is_train = True
		
		# build cuda accelerated RNN
		numHidden = 256
		numLayers = 2
		
		rnnIn3d = tf.transpose(rnnIn3d, [1, 0, 2])
		blstm = tf.contrib.cudnn_rnn.CudnnLSTM(numLayers, numHidden, direction='bidirectional')
		blstm.build(rnnIn3d.get_shape())
		
		# BxTxF -> BxTx2H
		blstm_output, _ = blstm(rnnIn3d, training=self.bool_is_train)
		print(blstm_output.get_shape())
		
		# BxTx2H -> BxTx1x2H
		blstm_output_expanded = tf.expand_dims(blstm_output, 2)
		print(blstm_output_expanded.get_shape())
		
		# project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
		print('kernel', [1, 1, numHidden * 2, len(self.charList) + 1])
		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=blstm_output_expanded, filters=kernel, rate=1, padding='SAME'), axis=[2])
		
		# Apply Language Model
		if self.languageModel:
			self.batch_cond_prob_tensor =  tf.log(self.applyLM(self.rnnOut3d))
			# do not compute gradient for conditional probabilities
			self.batch_cond_prob_tensor =  tf.stop_gradient(self.batch_cond_prob_tensor)
			self.rnnOut3d = tf.add(self.rnnOut3d, self.batch_cond_prob_tensor)
		
		self.rnnOut3d = tf.transpose(self.rnnOut3d, [1, 0, 2])
		print(self.rnnOut3d.get_shape())


	def setupCTC(self):
		"create CTC loss and decoder and return them"
		# BxTxC -> TxBxC
		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
		# ground truth text as sparse tensor
		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))
		self.gtLengths = tf.placeholder(tf.int32, shape=[None])
		#self.dense_labels = tf.placeholder(tf.int32, shape=[None, None])
		
		self.seqLen = tf.placeholder(tf.int32, [None])
		
		# decoder: either best path decoding or beam search decoding
		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)
		elif self.decoderType == DecoderType.WordBeamSearch:
			# import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
			word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

			# prepare information about language (dictionary, characters in dataset, characters forming words) 
			chars = str().join(self.charList)
			wordChars = open('../model/wordCharList.txt').read().splitlines()[0]
			corpus = open('../data/corpus.txt').read()

			# decode using the "Words" mode of word beam search
			self.decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(self.ctcIn3dTBC, dim=2), 50, 'Words', 0.0, corpus.encode('utf8'), chars.encode('utf8'), wordChars.encode('utf8'))
		
		
		# calc loss for batch
		self.ctcloss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True)
		
		if self.modifiedLossFunction:
			self.edit_dist = tf.edit_distance(self.decoder[0][0], tf.cast(self.gtTexts, dtype=tf.int64), normalize=False)
			# do not compute gradient for edit distance (treat as scaling factor)
			self.edit_dist = tf.stop_gradient(self.edit_dist)
			self.loss = tf.reduce_mean(tf.multiply(self.ctcloss, self.edit_dist + 1))
		else :
			self.loss = tf.reduce_mean(self.ctcloss)

		# calc loss for each element to compute label probability
		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)


	def setupTF(self):
		"initialize TF"
		print('Python: '+sys.version)
		print('Tensorflow: '+tf.__version__)
		
		config = tf.ConfigProto(log_device_placement=True)
		config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
		sess=tf.Session(config=config) # TF session

		saver = tf.train.Saver(max_to_keep=1) # saver saves model to file
		modelDir = '../model/'
		latestSnapshot = tf.train.latest_checkpoint(modelDir) # is there a saved model?

		# if model must be restored (for inference), there must be a snapshot
		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in: ' + modelDir)

		# load saved model if available
		if latestSnapshot:
			print('Init with stored values from ' + latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess,saver)
	
	
	def setupLM(self, wordsList):
		numOfChars = len(self.charList)
		if wordsList == None:
			equiprobable = np.ones([numOfChars+1, numOfChars+1])/(numOfChars+1)
			self.syllableProbs = tf.convert_to_tensor(equiprobable, dtype=tf.float32)
			return
		
		# count syllable hits (for given character in row, how many times the character in column has appeared)
		syllableHits = np.zeros([numOfChars, numOfChars])
		# count instances of the given characters
		characterHits = np.zeros([numOfChars])
		
		# count syllable appearance frequency
		for word in wordsList:
			prevIndex = -1
			for c in word:
				currIndex = self.charList.index(c)
				if prevIndex != -1:
					syllableHits[prevIndex][currIndex] += 1
					characterHits[prevIndex] += 1
				prevIndex = currIndex
		
		# avoid division by zero
		characterHits[characterHits == 0] = 1
		
		probs = np.divide(syllableHits, characterHits[:, None])
		# avoid -inf when used later in logarithm
		probs = probs + 0.0000001
		
		# extend conditional probability matrix to include blanks (with p =  1)
		extended_probs = np.ones([numOfChars+1, numOfChars+1])
		extended_probs[:-1, :-1] = probs
		
		# conditional probability tensor for all characters (given first character, what is the probability of the second character)
		self.syllableProbs = tf.convert_to_tensor(extended_probs, dtype=tf.float32)

	
	def applyLM(self, input_probs):
		# find max probability character for each time step
		max_prob_chars = tf.argmax(input_probs, axis=2)
		# extra "time step" column is added in front so conditional probability will not affect the first time step
		extra_timestep = tf.fill([len(self.charList), Model.batchSize], tf.cast(len(self.charList)+1, tf.int64))
		extended_max_prob_chars = tf.concat([extra_timestep, max_prob_chars], 0)
		
		# for each label, construct a tensor with appropriate probability "columns" 
		# (i.e. the "columns" corresponding to each previous character)
		cond_prob_vectors = []
		for b in range(0, Model.batchSize):
			vector = []
			prevCharIdx = len(self.charList)
			for i in range(0, self.maxTextLen):
				if self.sess:
					charIdx = extended_max_prob_chars[i][b].eval()
				else:
					charIdx = len(self.charList)
				
				if charIdx == len(self.charList):
					charIdx = prevCharIdx
				else:
					prevCharIdx = charIdx
				# get the coditional probabilities tensor (slice) of the given (previous) character
				cond_prob_matrix_slice = tf.slice(self.syllableProbs, [charIdx, 0], [1, -1])
				cond_prob_matrix_slice = tf.squeeze(cond_prob_matrix_slice, axis=0)
				vector.append(cond_prob_matrix_slice)
			# construct conditinal probabilities tensor for each label from list of slices
			cond_prob_tensor = tf.stack(vector, axis=0)
			cond_prob_vectors.append(cond_prob_tensor)
		
		# construct conditinal probabilities tensor of the batch
		return tf.stack(cond_prob_vectors, axis=1)
		

	def toSparse(self, texts):
		"put ground truth texts into sparse tensor for ctc_loss"
		indices = []
		values = []
		shape = [len(texts), 0] # last entry must be max(labelList[i])
		lengths = []
		
		# go over all texts
		for (batchElement, text) in enumerate(texts):
			# convert to string of label (i.e. class-ids)
			labelStr = [self.charList.index(c) for c in text]
			lengths.append(len(labelStr))
			# sparse tensor must have size of max. label-string
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			# put each label into sparse tensor
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape), (lengths)


	def decoderOutputToText(self, ctcOutput, batchSize):
		"extract texts from output of CTC decoder"
		
		# contains string of labels for each batch element
		encodedLabelStrs = [[] for i in range(batchSize)]

		# word beam search: label strings terminated by blank
		if self.decoderType == DecoderType.WordBeamSearch:
			blank=len(self.charList)
			for b in range(batchSize):
				for label in ctcOutput[b]:
					if label==blank:
						break
					encodedLabelStrs[b].append(label)

		# TF decoders: label strings are contained in sparse tensor
		else:
			# ctc returns tuple, first element is SparseTensor 
			decoded=ctcOutput[0][0] 

			# go over all indices and save mapping: batch -> values
			#idxDict = { b : [] for b in range(batchSize) }
			for (idx, idx2d) in enumerate(decoded.indices):
				label = decoded.values[idx]
				batchElement = idx2d[0] # index according to [b,t]
				encodedLabelStrs[batchElement].append(label)

		# map labels to chars for all batch elements
		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		"feed a batch into the NN to train it"
		numBatchElements = len(batch.imgs)
		sparse, sparse_lengths = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 7500 else (0.0001 if self.batchesTrained < 12500 else 0.00001)) # decay learning rate
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse, self.gtLengths: sparse_lengths, self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		"feed a batch into the NN to recognize the texts"
		
		# decode, optionally save RNN output
		numBatchElements = len(batch.imgs)
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if calcProbability or self.log else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)
		
		# Log CTC input probabilities, for each sample, in /src/data.txt
		if self.log:
			with open("../model/scores log.txt", "a") as f:
				np.set_printoptions(threshold=sys.maxsize)
				np.set_printoptions(suppress=True)
				score = np.transpose(evalRes[1], [1, 0, 2])
				for i in range(0, len(texts)):
					f.write(texts[i]+", ")
					f.write(batch.gtTexts[i]+"\n")
					f.write("      ")
					for c in self.charList:
						f.write(c+"        ")
					f.write("(blank)\n")
					f.write(np.array2string(score[i], precision=3, separator=', ', max_line_width=1024)+"\n\n")
				f.close()
		
		
		# feed RNN output and recognized text into CTC loss to compute labeling probability
		probs = None
		if calcProbability:
			sparse, sparse_lengths = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.gtLengths: sparse_lengths, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)
		return (texts, probs)
	

	def save(self):
		"save model to file"
		self.snapID += 1
		self.saver.save(self.sess, '../model/snapshot', global_step=self.snapID)
 
