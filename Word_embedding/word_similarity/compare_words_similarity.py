"""
# Reference


https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/


The basic idea of word embedding is words that occur in similar context tend to be closer to each other in vector space. For generating word vectors in Python, 
modules needed are nltk and gensim. Run these commands in terminal to install nltk and gensim :
- pip install nltk
- pip install gensim
"""

# Python program to generate word vectors using Word2Vec

# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
from IPython import get_ipython

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

# Reads 'alice.txt' file
sample = open("./data/alice.txt", encoding="utf8")
s = sample.read()

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in sent_tokenize(f):
	temp = []

	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

# Create CBOW model
"""
Note that the input data is alway list of list. In this case, we read the text from the file, then do the following things:
- Use NLTK to convert text to sentence
- Use NLTK to covert sentence to words
Then the input data will be the list of list of words.
"""

model1 = Word2Vec(data, min_count = 1,vector_size = 100, window = 5)

# Print results
print("Cosine similarity between 'alice' " +
			"and 'wonderland' - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
				"and 'machines' - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,
											window = 5, sg = 1)

# Print results
print("Cosine similarity between 'alice' " +
		"and 'wonderland' - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
			"and 'machines' - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))


"""
Online training / Resuming training
Advanced users can load a model and continue training it with more sentences and new vocabulary words:
"""
more_sentences = ["Alice is a good girl", "Alice like wearing skirt", "Alice love going to the cinema alone", "wonderland actually a hell where human work for machine", "machine will destroy all human"]
more_sentences_token = []

for i in more_sentences:
	temp = []

	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())

	more_sentences_token.append(temp)

model1.build_vocab(more_sentences_token, update=True)
model1.train(more_sentences_token, total_examples=model1.corpus_count, epochs=model1.epochs)

model2.build_vocab(more_sentences_token, update=True)
model2.train(more_sentences_token, total_examples=model2.corpus_count, epochs=model2.epochs)

# Print results
print("Cosine similarity between 'alice' " +
			"and 'wonderland' after update training - CBOW : ",
	model1.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
				"and 'machines' after update training  - CBOW : ",
	model1.wv.similarity('alice', 'machines'))

print("Cosine similarity between 'alice' " +
		"and 'wonderland' after update training - Skip Gram : ",
	model2.wv.similarity('alice', 'wonderland'))
	
print("Cosine similarity between 'alice' " +
			"and 'machines' after update training - Skip Gram : ",
	model2.wv.similarity('alice', 'machines'))


"""
You can see that the score after update training is slighty different(because we didn't add too much data)
"""



"""
Visualising Word Embeddings

The word embeddings made by the model can be visualised by reducing dimensionality of the words to 2 dimensions using tSNE.

Visualisations can be used to notice semantic and syntactic trends in the data.

Example:
- Semantic: words like cat, dog, cow, etc. have a tendency to lie close by
- Syntactic: words like run, running or cut, cutting lie close together.
"""


from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np                                  # array handling
import matplotlib.pyplot as plt
import random

def reduce_dimensions(model):
    num_dimensions = 3  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


x_vals, y_vals, labels = reduce_dimensions(model1)



random.seed(0)
plt.figure(figsize=(12, 12))
plt.scatter(x_vals, y_vals)

#
# Label randomly subsampled 25 data points
#
indices = list(range(len(labels)))
selected_indices = random.sample(indices, 25)
for i in selected_indices:
	plt.annotate(labels[i], (x_vals[i], y_vals[i]))
plt.show()



