# NLP Interview Questions
A collection of technical interview questions for NLP  positions.

Some popular github :
- https://github.com/khangich/machine-learning-interview


#### 1) What is Word embedding
Word embedding is vector representations of a particular word. Word2Vec is a method to construct such an embedding, It can be obtained using two methods
- CBOW Model : This method takes the context of each word as the input and tries to predict the word corresponding to the context
- Skip-gram model : We can use the target word (whose representation we want to generate) to predict the context and in the process, we produce the representations

https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa

#### 2) seq2seq model 

use mainly for translation
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

#### 3) Explain transformer 

Transformer including input, encoders, decoders, output. The encoding components is a stack of encoders, The decoding component is a stack of decoders of the same number.

https://jalammar.github.io/illustrated-transformer/


#### 4) Explain the difference between gensim and bert 
- gensim is library to implement word2vec
- transformer is libary which has the model bert

#### 5.0 How RNN work ?

Take a look at the notebook to see the theory and code implementation(Didn't )


#### 5.0 What is the transformer encoder ?

Reference : http://cs231n.stanford.edu/slides/2021/lecture_11.pdf

#### 5) How BERT work, why do you want to use BERT

BERT architecture consists of several Transformer encoders stacked together. Each Transformer encoder encapsulates two sub-layers: a self-attention layer and a feed-forward layer. Bert convert text to vector by tokenization then start training on pretrained model(Bert base or bert large).

There are at least two reasons why BERT is a powerful language model:

- It is pre-trained on unlabeled data extracted from BooksCorpus, which has 800M words, and from Wikipedia, which has 2,500M words.
- As the name suggests, it is pre-trained by utilizing the bidirectional(*) nature of the encoder stacks. This means that BERT learns information from a sequence of words not only from left to right, but also from right to left 

reference : 
- https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
- '*' : chrome-extension://abkfbakhjpmblaafnpgjppbmioombali/pdfjs/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fseminar%2Fdetails%2Fjdevlin.pdf

(need to undestand deeper)
#### 6) Explain what is semantic analysis

semantic analysis including Word Sense Disambiguation and Relationship Extraction, in order to understand the meaning of a sentence.

https://www.geeksforgeeks.org/understanding-semantic-analysis-nlp/






