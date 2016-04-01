# object-tagging-in-vqa-dataset
A word2vec model is trained on the brown corpus, MS COCO caption dataset and VQA question dataset.
Then the embeddings are used for training a system on the Brown corpus to categorize words on the basis of their POS tag.
The POS tagging (or more precisely, 'noun'/'not a noun' tagging is done both on individual word embeddings and on a sliding window of word embeddings.
