import gensim, logging
import numpy as np
from nltk.corpus import brown
import h5py
import re
from datetime import datetime
from keras.utils import np_utils


# print 'Loading model and dataset...'
# new_model=gensim.models.Word2Vec.load('./vqa_train')
# tag_sent=brown.tagged_sents()

# startTime = datetime.now()
# count=2

# words=[]

# print 'Converting to a 1-D list'
# for sent in tag_sent:
# 	for word in sent:
# 		words.append(word)

# del tag_sent

# print 'Preparing vectors...'
# X_train=new_model[words[0][0]].astype('float16')
# prog=re.compile('^NN(.)*$')
# result=prog.match(words[0][1])
# if result:
# 	Y_train=np.array([1])
# else:
# 	Y_train=np.array([0])


# for word in words[1:]:
# 	vec=new_model[word[0]].astype('float16')
# 	result=prog.match(word[1])

# 	print count
# 	X_train=np.vstack((X_train,vec))
# 	if result:
# 		Y_train=np.concatenate((Y_train,[1]))
# 	else:
# 		Y_train=np.concatenate((Y_train,[0]))
# 	count+=1

# print 'Saving vectors...'
# print X_train.shape
# print Y_train.shape

# with h5py.File('pos.h5', 'w') as hf:
#     hf.create_dataset('X_train', data=X_train)
#     hf.create_dataset('Y_train', data=Y_train)

# print 'Time taken: '+str(datetime.now() - startTime)


def chunks(l, n):
	"""Yield successive n-sized chunks from l"""
	for i in xrange(0, len(l), n):
		yield l[i:i+n]

def datasetGenerator_classification(word_list,new_model,tagset):
	X_train=new_model[word_list[0][0]].astype('float16')
	Y_train=np.array([tagset[word_list[0][1]]])

	for word in word_list[1:]:
		vec=new_model[word[0]].astype('float16')
		X_train=np.vstack((X_train,vec))
		Y_train=np.concatenate((Y_train,[tagset[word[1]]]))
	Y_train = np_utils.to_categorical(Y_train, len(tagset))
	return X_train,Y_train



def datasetGenerator_regression(word_list,new_model):
	X_train=new_model[word_list[0][0]].astype('float16')
	prog=re.compile('^NN(.)*$')
	result=prog.match(word_list[0][1])
	if result:
		Y_train=np.array([1])
	else:
		Y_train=np.array([0])


	for word in word_list[1:]:
		vec=new_model[word[0]].astype('float16')
		result=prog.match(word[1])

		X_train=np.vstack((X_train,vec))
		if result:
			Y_train=np.concatenate((Y_train,[1]))
		else:
			Y_train=np.concatenate((Y_train,[0]))
	return X_train,Y_train

def datasetGenerator_slidingWindow_regression(word_list,new_model,window_size):
	middle=(window_size-1)/2
	slides=len(word_list)-window_size-1

	window=word_list[0:window_size]
	vec=new_model[window[0][0]].astype('float16')
	for word in window[1:]:
		vec=np.append(vec,new_model[word[0]].astype('float16'))
	prog=re.compile('^NN(.)*$')
	result=prog.match(window[middle][1])
	X_train=vec
	if result:
		Y_train=np.array([1])
	else:
		Y_train=np.array([0])


	for i in xrange(1,slides):
		window=word_list[i:i+window_size]
		vec=new_model[window[0][0]].astype('float16')
		for word in window[1:]:
			vec=np.append(vec,new_model[word[0]].astype('float16'))
		prog=re.compile('^NN(.)*$')
		result=prog.match(window[middle][1])
		X_train=np.vstack((X_train,vec))
		if result:
			Y_train=np.concatenate((Y_train,[1]))
		else:
			Y_train=np.concatenate((Y_train,[0]))

	return X_train,Y_train

def loadData():
	new_model=gensim.models.Word2Vec.load('./vqa_train')
	words=brown.tagged_words(tagset='universal')

	i=0
	tagset={}
	for word in words:
		if word[1] in tagset:
			pass
		else:
			tagset[word[1]]=i
			i+=1
	return words,new_model,tagset

def getTestData_regression(words,new_model,window_size):
	test_words=words[-1000:]
	return datasetGenerator_regression(test_words,new_model)
	# return datasetGenerator_slidingWindow_regression(words,new_model,window_size)

def getTestData_classification(words,new_model,window_size,tagset):
	test_words=words[-1000:]
	return datasetGenerator_classification(test_words,new_model,tagset)
	# return datasetGenerator_slidingWindow_regression(words,new_model,window_size)