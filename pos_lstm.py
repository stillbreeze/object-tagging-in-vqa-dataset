import numpy as np
import prep_data
import h5py

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM


def test_regression(model,X_test,Y_test):
	# print 'Loading weights...'
	# model.load_weights('pos.h5')
	print 'Testing model...'
	pred = model.predict(X_test)
	pred=pred.flatten()
	predictions=(pred>0.1).astype('int')
	res=np.logical_xor(predictions,Y_test).astype('int')
	res=np.sum(res)
	res=1000-res
	print predictions[0:50]
	print '--------------------------------------------------------'
	print Y_test[0:50]
	print '\nTest Accuracy:\t' + str(float(res)/1000) + '\n\n'

def test_classification(model,X_test,Y_test):
	# print 'Loading weights...'
	# model.load_weights('pos.h5')
	print 'Testing model...'
	correct=0
	pred = model.predict(X_test)
	for predicted,answer in zip(pred,Y_test):
		if np.argmax(predicted)==np.argmax(answer):
			correct+=1
	print pred[0:50]
	print '--------------------------------------------------------'
	print Y_test[0:50]
	print '\nTest Accuracy:\t' + str(float(correct)/1000) + '\n\n'

def RNN():
	print 'Loading data...'
	words,new_model,tagset=prep_data.loadData()

	maxlen = 200
	nb_classes=len(tagset)
	batch_size = 128
	nb_epoch=200
	chunk_len=2000
	window_size=5

	nb_filter=64
	filter_length=3
	pool_length=2

	X_test,Y_test=prep_data.getTestData_classification(words,new_model,window_size,tagset)

	print('Building model...')
	model = Sequential()
	model.add(Embedding(batch_size, maxlen, dropout=0.5))
	model.add(Convolution1D(nb_filter=nb_filter,
						filter_length=filter_length,
						border_mode='valid',
						activation='relu',
						subsample_length=1))
	model.add(MaxPooling1D(pool_length=pool_length))
	# model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1,return_sequences=True,init='he_normal'))
	model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1,init='he_normal'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	# model.add(Reshape((maxlen,1),input_shape=(maxlen,)))
	# model.add(LSTM(256,batch_input_shape=(batch_size, maxlen, 1),return_sequences=True,dropout_W=0.5, dropout_U=0.1))
	# model.add(LSTM(256,batch_input_shape=(batch_size, maxlen, 1),return_sequences=False,dropout_W=0.5, dropout_U=0.1))
	# model.add(Dense(1))
	# model.add(Dropout(0.5))
	# model.add(Activation('sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='adam')

	print 'Starting with training...'
	for e in range(nb_epoch):
		chunk_count=0
		print("epoch %d" % e)
		for word_list in prep_data.chunks(words,chunk_len):
			chunk_count+=1
			print ("chunk %d" % chunk_count)
			X_batch,Y_batch=prep_data.datasetGenerator_classification(word_list,new_model,tagset)
			model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1,validation_split=0.1,show_accuracy=True)
			test_classification(model,X_test,Y_test)
			if chunk_count%10==0:
				model.save_weights('pos2.h5',overwrite=True)
			if chunk_count==100:
				break

if __name__ == "__main__":
	RNN()