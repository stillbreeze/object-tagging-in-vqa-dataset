import gensim, logging
import numpy as np
from nltk.corpus import brown
import h5py
import re
from datetime import datetime


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

def datasetGenerator(word_list,new_model):
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

def datasetGenerator_slidingWindow(word_list,new_model,window_size):
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

	print X_train.shape
	print Y_train.shape

	return X_train,Y_train

def loadData():
	new_model=gensim.models.Word2Vec.load('./vqa_train')
	tag_sent=brown.tagged_sents()

	words=[]
	for sent in tag_sent:
		for word in sent:
			words.append(word)

	del tag_sent
	return words,new_model