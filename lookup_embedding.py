import gensim
from nltk.corpus import brown
import h5py
import numpy as np

new_model=gensim.models.Word2Vec.load('./vqa_train')
words=brown.tagged_words(tagset='universal')

index_dict={}
word_vec={}
t=1

print 'indexing vocabulary'
for i in words:
	if i[0] not in index_dict:
		index_dict[i[0]]=t
		t+=1

print 'creating embedding weights'
vocab_dim = 200
n_symbols = len(index_dict) + 1
embedding_weights = np.zeros((n_symbols+1,vocab_dim))
for word,index in index_dict.items():
    embedding_weights[index,:] = new_model[word]


print 'writing to file'
with h5py.File('embedding_weights.h5', 'w') as hf:
	hf.create_dataset('embedding_weights', data=embedding_weights)