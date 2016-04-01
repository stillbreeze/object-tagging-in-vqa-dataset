import numpy as np
import prep_data
import h5py

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


def RNN():
	words,new_model=prep_data.loadData()
	maxlen = 200
	batch_size = 128
	nb_epoch=200
	chunk_len=10000
	window_size=5

	print('Building model...')
	model = Sequential()
	model.add(Embedding(batch_size, maxlen, dropout=0.5))
	# model.add(LSTM(256, dropout_W=0.5, dropout_U=0.1,return_sequences=True))
	model.add(LSTM(128, dropout_W=0.5, dropout_U=0.1))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	# model.add(Reshape((maxlen,1),input_shape=(maxlen,)))
	# model.add(LSTM(256,batch_input_shape=(batch_size, maxlen, 1),return_sequences=True,dropout_W=0.5, dropout_U=0.1))
	# model.add(LSTM(256,batch_input_shape=(batch_size, maxlen, 1),return_sequences=False,dropout_W=0.5, dropout_U=0.1))
	# model.add(Dense(1))
	# model.add(Dropout(0.5))
	# model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='adam')

	print 'Starting with training...'
	for e in range(nb_epoch):
		chunk_count=0
		print("epoch %d" % e)
		for word_list in prep_data.chunks(words,chunk_len):
			chunk_count+=1
			print ("chunk %d" % chunk_count)
			X_batch,Y_batch=prep_data.datasetGenerator_slidingWindow(word_list,new_model,window_size)
			model.fit(X_batch, Y_batch, batch_size=batch_size, nb_epoch=1,validation_split=0.2,show_accuracy=True)
			if chunk_count==80:
				model.save_weights('pos.h5')
				break

if __name__ == "__main__":
	RNN()