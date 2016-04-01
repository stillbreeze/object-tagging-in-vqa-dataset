import nltk
import json
import os
from nltk.corpus import brown
def getSentences():
	print '-------------------------------------------------'
	sentences=[]
	corpus=brown.sents()

	print 'reading Brown corpus...'
	for i in corpus:
		sentences.append(i)

	print 'reading COCO corpus...'
	with open('captions_train2014.json') as inp:
		j = json.load(inp)

	captions=j['annotations']

	for caption in captions:
		text = nltk.word_tokenize(caption['caption'])
		sentences.append(text)

	with open('captions_val2014.json') as inp:
		j = json.load(inp)

	captions=j['annotations']

	for caption in captions:
		text = nltk.word_tokenize(caption['caption'])
		sentences.append(text)


	print 'reading VQA corpus...'
	with open('OpenEnded_mscoco_train2014_questions.json') as inp:
		j = json.load(inp)

	questions=j['questions']

	for question in questions:
		text = nltk.word_tokenize(question['question'])
		sentences.append(text)

	with open('OpenEnded_mscoco_val2014_questions.json') as inp:
		j = json.load(inp)

	questions=j['questions']

	for question in questions:
		text = nltk.word_tokenize(question['question'])
		sentences.append(text)

	print '-------------------------------------------------'

	return sentences