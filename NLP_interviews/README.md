# NLP Interview Questions
A collection of technical interview questions for NLP  positions.

Some popular github :
- https://github.com/khangich/machine-learning-interview


#### What is transfer learning 

When someone else creates a model on a huge generic dataset and passes only the model to others for use. This is known as transfer learning because everyone doesn’t have to train the model on such a huge amount of data, hence, they “transfer” the learnings from others to their system.

Transfer learning is really helpful in NLP. Specially vectorization of text, because converting text to vectors for 50K records also is slow. So if we can use the pre-trained models from others, that helps to resolve the problem of converting the text data to numeric data, and we can continue with the other tasks, such as classification or sentiment analysis, etc.


#### 1) What is Word embedding

Take a look at the lab of X

Word embedding is vector representations of a particular word. Word2Vec is a method to construct such an embedding, It can be obtained using two methods

- CBOW Model : This method takes the context of each word as the input and tries to predict the word corresponding to the context(See example in the reference)

More detailed :
* https://thinkinfi.com/continuous-bag-of-words-cbow-single-word-model-how-it-works/
* https://thinkinfi.com/continuous-bag-of-words-cbow-multi-word-model-how-it-works/

- Skip-gram model : basically the inverse of the CBOW model. We can use the target word to predict the context and in the process, we produce the representations (see the example in the reference). 
More detailed :
* http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
Generally, the skip-gram method can have a better performance compared with CBOW method, for it can capture two semantics for a single word. For instance, it will have two vector representations for Apple, one for the company and another for the fruit. 
For more details about the word2vec algorithm, please check here.

https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa


#### 2> Word2Vec is supervised learning or unsupervised learning ?

It's Unsupervised, but It's more like self supervised because it label the data it self

Long Story:

Word2Vec uses a neural network to form word embeddings.

A neural network works by back-propagating error.

To calculate error, you need labelled data.

Word2Vec reads the text and generates labelled data from it.

Consider the input text : ‘I like chicken wings’

Word2Vec’s pre-processing part takes this sentence and makes data pairs like:

( I, like)
( like, I), ( like, chicken)
( chicken, like), (chicken, wings)
( wings, chicken)
Considering a window size of 1 (i.e., for each word, you look to the 1 to theleft and one to the right)

So when you input ‘chicken ‘ into the network, it will try to predict the closest word as ‘like’ or ‘wings’.

Thus it forms labels and trains the network.

Word2Vec is not a true unsupervised learning technique (since there is some sort of error backpropagation taking place through correct and incorrect predictions), they are a self-supervised technique, a specific instance of supervised learning where the targets are generated from the input data. In order to get self-supervised models to learn interesting features, you have to come up with an interesting synthetic target and loss function.

#### How to evaluate word2vec :

Word2Vec training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application.

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

The main difference between Feed Forward Neural Network and ss RNN is :

Traditional feed-forward neural networks take in a fixed amount of input data all at the same time and produce a fixed amount of output each time. On the other hand, RNNs do not consume all the input data at once. nstead, they take them in one at a time and in a sequence. At each step, the RNN does a series of calculations before producing an output. The output, known as the hidden state, is then combined with the next input in the sequence to produce another output. This process continues until the model is programmed to finish or the input sequence ends.

Reference :
- https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


#### 5.0 What is the transformer encoder ?
The transformer uses an encoder-decoder architecture. The encoder extracts features from an input sentence, and the decoder uses the features to produce an output sentence (translation).
- The encoder in the transformer consists of multiple encoder blocks. An input sentence goes through the encoder blocks, and the output of the last encoder block becomes the input features to the decoder.
- The decoder also consists of multiple decoder blocks.

https://kikaben.com/transformers-encoder-decoder/
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






