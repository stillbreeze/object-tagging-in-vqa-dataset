import gensim, logging
from get_sentences import getSentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print 'getting sentences...'
sentences=getSentences()

print 'training word2vec on ' + str(len(sentences)) + ' sentences...'
model = gensim.models.Word2Vec(sentences, min_count=1,workers=4,size=200)

print 'saving model'
model.save('./vqa_train')