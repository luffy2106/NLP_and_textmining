# Theory

Word embedding is vector representations of a particular word. Word2Vec is a method to construct such an embedding, It can be obtained using two methods:
- CBOW Model : This method takes the context of each word as the input and tries to predict the word corresponding to the context(See example in the reference)

More detailed :
* https://thinkinfi.com/continuous-bag-of-words-cbow-single-word-model-how-it-works/
* https://thinkinfi.com/continuous-bag-of-words-cbow-multi-word-model-how-it-works/

- Skip-gram model : basically the inverse of the CBOW model. We can use the target word to predict the context and in the process, we produce the representations (see the example in the reference). 
More detailed :
* http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

Application of word embedding:
- Sentiment Analysis
- Speech Recognition
- Information Retrieval
- Question Answering