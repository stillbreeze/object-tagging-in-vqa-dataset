# object-tagging-in-vqa-dataset
The final aim is to identify the presence of common objects in sentences, like that in the VQA Questions Dataset.
It's a refinement of the POS tagging problem.

A word2vec model is trained on the Brown corpus, MS COCO caption dataset and VQA question dataset.
Then the embeddings are used for training a system on the Brown corpus to categorize words on the basis of their POS tag.
The POS tagging (or more precisely, 'noun'/'not a noun' tagging is done both on individual word embeddings and on a sliding window of word embeddings.

Extra todo :Eliminate abstract and verbal nouns to obtain only concrete nouns (objects)
