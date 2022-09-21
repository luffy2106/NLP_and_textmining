# Theory

Word embedding is vector representations of a particular word. Word2Vec is a method to construct such an embedding, It can be obtained using two methods:
- CBOW Model : This method takes the context of each word as the input and tries to predict the word corresponding to the context(See example in the reference)

More detailed :
* https://thinkinfi.com/continuous-bag-of-words-cbow-single-word-model-how-it-works/
* https://thinkinfi.com/continuous-bag-of-words-cbow-multi-word-model-how-it-works/

- Skip-gram model : basically the inverse of the CBOW model. We can use the target word to predict the context and in the process, we produce the representations (see the example in the reference). 
More detailed :
* http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
Generally, the skip-gram method can have a better performance compared with CBOW method, for it can capture two semantics for a single word. For instance, it will have two vector representations for Apple, one for the company and another for the fruit. 
For more details about the word2vec algorithm, please check here.


Application of word embedding:
- Sentiment Analysis
- Speech Recognition
- Information Retrieval
- Question Answering


# Gensim

Gensim is a libary to implement word2vec, doc2vec
- Word2Vec is a Model that represents each Word as a Vector.
- Doc2Vec is a Model that represents each Document as a Vector.

# Word2vec

The input of gensim word2vec always list of list, besides, these parameters are important:
- size: The number of dimensions of the embeddings and the default is 100.
- window: The maximum distance between a target word and words around the target word. The default window is 5.
- min_count: The minimum count of words to consider when training the model, words with occurrence less than this count will be ignored. 
The default for min_count is 5.


# Problem word embedding can solve

- Compare the similarity between 2 sentences
- Word embeddings for text classification
- Knn classifier to categorize documents
