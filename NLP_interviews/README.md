# NLP Interview Questions
A collection of technical interview questions for NLP  positions.

Some popular github :
- https://github.com/khangich/machine-learning-interview


#### What is transfer learning 

When someone else creates a model on a huge generic dataset and passes only the model to others for use. This is known as transfer learning because everyone doesn’t have to train the model on such a huge amount of data, hence, they “transfer” the learnings from others to their system.

Transfer learning is really helpful in NLP. Specially vectorization of text, because converting text to vectors for 50K records also is slow. So if we can use the pre-trained models from others, that helps to resolve the problem of converting the text data to numeric data, and we can continue with the other tasks, such as classification or sentiment analysis, etc.


#### What is Word embedding

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

Choice of model architecture
- Large corpus, higher dimensions, slower - Skip gram
- Small corpus, faster, cbow

#### Word embedding category 

There are several word embedding methods which can be divided into two major categories : 𝗖𝗼𝗻𝘁𝗲𝘅𝘁-𝗶𝗻𝗱𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁 and 𝗖𝗼𝗻𝘁𝗲𝘅𝘁-𝗱𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁

✅ Context-independent methods are characterized by being unique and distinct for each word without considering the word’s context.

🔹 Context-independent without machine learning

𝗕𝗮𝗴-𝗼𝗳-𝘄𝗼𝗿𝗱𝘀: This method represents a text, such as a sentence or a document, as the bag of its words, disregarding grammar and even word order but keeping multiplicity.

𝗧𝗙-𝗜𝗗𝗙: This gets this importance score by getting the term’s frequency (TF) and multiplying it by the term inverse document frequency (IDF).

🔹 Context-independent with machine learning

𝗪𝗼𝗿𝗱𝟮𝗩𝗲𝗰: A shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words, utilize either of two model architectures: continuous bag-of-words (CBOW) or continuous skip-gram.

𝗚𝗹𝗼𝗩𝗲: This performs training on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

𝗙𝗮𝘀𝘁𝗧𝗲𝘅𝘁: This method embeds words by treating each word as being composed of character n-grams instead of a word whole. This feature enables it not only to learn rare words but also out-of-vocabulary words.

✅ Context-dependent learns different embeddings for the same word based on its context.

🔹 Context-dependent and RNN based

𝗘𝗟𝗠𝗢: learns contextualized word representations based on a neural language model with a character-based encoding layer and two BiLSTM layers.

𝗖𝗼𝗩𝗲: uses a deep LSTM encoder from an attentional sequence-to-sequence model trained for machine translation to contextualize word vectors.

🔹 Context-dependent and transformer-based

𝗕𝗘𝗥𝗧: This is a transformer-based language representation model trained on a large cross-domain corpus, which uses a masked language model to predict words that are randomly masked in a sequence.

𝗫𝗟𝗠: Another transformer based model which pretrained using next token prediction, masked language modeling and a translation objective.

𝗥𝗼𝗕𝗘𝗥𝗧𝗮: This is built on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.

𝗔𝗟𝗕𝗘𝗥𝗧: This is a parameter-reduction techniques to lower memory consumption and increase the training speed of BERT.

reference : 
https://www.linkedin.com/posts/puspanjalisarma_nlp-datascience-artificialintelligence-activity-6981164409419149313-fwH9?utm_source=share&utm_medium=member_desktop

![Alt text](picture/word_embedding_category.jpg?raw=true "Word embedding category")


#### How word2vec convert word to vetor ?
word2vec is neural network 
- For CBOW, the input is context of each word and the label is the word respoding to the context.
- For skip-gram, target word to predict the context.

Each word was convert to one hot vector. After trainning we have matrix weights. If we multiply matrix weight 
with one hot vector corresponding to word, we will have word vector.

see reference : https://www.youtube.com/watch?v=UqRCEmrv1gQ (6:24)


#### what is word2vec and doc2vec ?
- word2vec : algorithm based on neural networks, the input is a word and the output is a vector. Then we can compare
the similarity between words by consine similarity
- doc2vec : algorithm based on neural networks, the input is a document and the output is a vector. Then we can compare
the similarity between docs by consine similarity

#### 2> Word2Vec and doc2vec is supervised learning or unsupervised learning ?

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

#### How to evaluate word2vec/doc2vec :

Word2Vec/doc2vec training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application(human eyes)


#### 2) seq2seq model 

use mainly for translation
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

#### 3) Explain transformer 

Transformer including input, encoders, decoders, output. The encoding components is a stack of encoders, The decoding component is a stack of decoders of the same number.

https://jalammar.github.io/illustrated-transformer/


#### 4) Explain the difference between gensim and transformer 
- gensim is library to implement word2vec model
- transformer is libary to implement model bert

#### Explain the difference between word2vec model and bert model
- Word2Vec models generate embeddings that are context-independent: ie - there is just one vector (numeric) representation for each word. Different senses of the word (if any) are combined into one single vector.
- However, the BERT model generates embeddings that allow us to have multiple (more than one) vector (numeric) representations for the same word, based on the context in which the word is used. Thus, BERT embeddings are context-dependent.

For example, in the sentence:
- We went to the river bank.
- I need to go to bank to make deposit

the word 'bank' is being used in two different contexts 
- financial entity
- land along the river (geography).

Word2Vec will generate the same single vector for the word bank for both the sentences. Whereas, BERT will generate two different vectors for the word bank being used in two different contexts. One vector will be similar to words like money, cash etc. The other vector would be similar to vectors like beach, coast etc


#### 5.0 How RNN work ?

The main difference between Feed Forward Neural Network and ss RNN is :

Traditional feed-forward neural networks take in a fixed amount of input data all at the same time and produce a fixed amount of output each time. On the other hand, RNNs do not consume all the input data at once. Instead, they take them in one at a time and in a sequence. At each step, the RNN does a series of calculations before producing an output. The output, known as the hidden state, is then combined with the next input in the sequence to produce another output. This process continues until the model is programmed to finish or the input sequence ends.

Reference :
- https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


#### 5.0 What is the transformer encoder decoder ?(need to research in detailed)
The transformer uses an encoder-decoder architecture. The encoder extracts features from an input sentence, and the decoder uses the features to produce an output sentence (translation).
- The encoder in the transformer consists of multiple encoder blocks. An input sentence goes through the encoder blocks, and the output of the last encoder block becomes the input features to the decoder.
- The decoder also consists of multiple decoder blocks.

https://kikaben.com/transformers-encoder-decoder/
Reference : http://cs231n.stanford.edu/slides/2021/lecture_11.pdf

#### 5) How BERT work, why do you want to use BERT
(see popular_model repor to see how to implement it)
BERT architecture consists of several Transformer encoders stacked together. Each Transformer encoder encapsulates two sub-layers: a self-attention layer and a feed-forward layer. Bert convert text to vector by tokenization then start training on pretrained model(Bert base or bert large).
Basically Bert convert your input string to vector, with the following features:
- BERT base, which is a BERT model consists of 12 layers of Transforme
- maximum size of tokens that can be fed into BERT model is 512(the text should not have more than 512 words).
- Bert automatically add [CLS] and [SEP] to the begin and end of each string, and some padding if the length of text is shorter than the standard.
- The [cls] token is used for classification task(the model will do classification with the input string begin with [cls]) whereas the [sep] is used to indicate the end of every sentence
- BERT model then will output an embedding vector of size 768 in each of the tokens. We can use these vectors as an input for different kinds of NLP applications, whether it is text classification, next sentence prediction, Named-Entity-Recognition (NER), or question-answering.

There are at least two reasons why BERT is a powerful language model:

- It is pre-trained on unlabeled data extracted from BooksCorpus, which has 800M words, and from Wikipedia, which has 2,500M words.
- As the name suggests, it is pre-trained by utilizing the bidirectional(*) nature of the encoder stacks. This means that BERT learns information from a sequence of words not only from left to right, but also from right to left 

reference : 
- https://towardsdatascience.com/text-classification-with-bert-in-pytorch-887965e5820f
- '*' : chrome-extension://abkfbakhjpmblaafnpgjppbmioombali/pdfjs/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fseminar%2Fdetails%2Fjdevlin.pdf

How to evaluate Bert ?

Since Bert only covert texts to vector, so the evaluation depend on the appplication. For classification, we take bert output of 
size 768 then add linear layer with ReLU activation function. At the end of the linear layer, we have a vector of size 5, each corresponds to a category of our labels (sport, business, politics, entertainment, and tech).

(need to undestand deeper)
#### 6) Explain what is semantic analysis

semantic analysis including Word Sense Disambiguation and Relationship Extraction, in order to understand the meaning of a sentence.

https://www.geeksforgeeks.org/understanding-semantic-analysis-nlp/

#### Suppose that you use doc2vec to compare the similarity of the requirements in your project, how could you evaluate your model ?

- We ask our client to label which requirements are similar and put to a same cluster. Suppose that we have 4 clusters
- we embedding all the requirements to vecto using doc2vec.
- Now we convert unsuppervised learning to suppervised learning by try to predict if a requirement belong to a cluster.
With the converted vector, we can use logistic regression, KNN, adaboost or neural network(if string have a lot of dimensions) to train, test and evaluate.
- After doing prediction, we can see how much requirements in the same cluster similar by compare the similarity of 2 vectors.


#### How TFIDF work ?

The overall goal of TF-IDF is to statistically measure how important a word is in a collection of documents. It's like a really useful keyword density tool on steroids.	

The TF-IDF weight computation is based on the product of two separate factors, namely the Term Frequency (TF) and the Inverse
Document Frequency (IDF). The intuition behind this measure, is that a term (word) is very important if it appears many times inside a document AND the number of documents that the this term is present, is relatively small.

the importance (i.e., the weight) of a term t in document d, is quantified by:
WEIGHT(t, d) = TF(t, d) * IDF(t)

Note:
* TF(t, d) = (number of occurrences of term t in doc d) / (number of words of d)
* IDF(t) = log (N/(1+Nt)), where N is the total number of docs and Nt the number of docs containing t

How TFIDF convert text to vector ?

TF-IDF vectorization involves calculating the TF-IDF score for every word in your corpus relative to that document and then putting that information into a vector (see link below using example documents “A” and “B”). Thus each document in your corpus would have its own vector, and the vector would have a TF-IDF score for every single word in the entire collection of documents. Once you have these vectors you can apply them to various use cases such as seeing if two documents are similar by comparing their TF-IDF vector using cosine similarity.

reference:
- https://www.capitalone.com/tech/machine-learning/understanding-tf-idf/


#### What is the difference between doc2vec tfidf, gensim doc2vec and bert
- doc2vec convert string and word to vector based on the relationship between word in "one context"
- bert convert string and word to vector based on the relationship between word in "different context"
- tfidf convert string and word to vector based on the "importance of the word in a document"


#### What is the difference between Bert and Sbert?

Bert is embedded in word level(each vector is a word) but S-bert is embedded in sentence level(each vector is a sentence).

Note that you can have vectors in sentence level in Bert but this sentence is the combination of words, you have to do another step to have vectors in sentence level.
On the other hand, a lot of pre-trained was done in Sbert, you can have a vector in Sentence level directly. Take a look at the example below to understand in detail:

```
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode a sentence
sentence = "This is a sample sentence."
tokens = tokenizer.encode(sentence, add_special_tokens=True)
outputs = model(torch.tensor([tokens]))[0]
sentence_embedding = outputs[0][0].detach().numpy()
```

In this example, the BertTokenizer class is used to tokenize the input sentence, and the BertModel class is used to obtain the sentence embedding. The encoding method of the tokenizer is used to add special tokens to the input sentence, and the BertModel instance is called with the encoded input to obtain the model outputs. The sentence embedding is then extracted from the outputs by taking the final hidden state corresponding to the [CLS] token.



#### Word prediction

We want to build a language model to predict what words come next. Which of the following architecture/model is best suitable ?
1. LSTM
2. BERT
3. bidirectional LTSM 
4. convolutional network

Bert and bidirectional LTSM utilize the bidirectional stack. It means that they learn information from a sequence of words not only from left to right, but also from right to left.
Therefore, the anwser is 1 - LSTM.

#### What are the most 3 relevant metrics to determine the similarity between 2 words/sentences with normalized embedding ?
1. Cosine
2. Jaccard
3. Euclidian
4. Hamming 
5. Dot Product

- Cosine similarity : convert 2 strings to vector and measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them(compare the angle)
```
https://www.geeksforgeeks.org/python-measure-similarity-between-two-sentences-using-cosine-similarity/
```
- Jaccard compare how two set (of n-grams) are similar : J(A, B) = (A∩B) / (A∪B)
```
https://stackoverflow.com/questions/39461813/using-jaccard-coefficient-for-measuring-string-similarity
```
- Euclidian distance : convert 2 string to vectors then compare the distance between 2 vectors(compare the distance)

- Hamming Distance between two strings(only apply for string with equal length).  the Hamming distance between two strings of equal length is the number of positions at which the corresponding character is different. 
```
https://www.geeksforgeeks.org/hamming-distance-two-strings/
```
- Dot product consider both cosine and the length of the vectors. Suppose there are 3 vector a,b,c . Even the cosine between vector a and b is higher than consine between vector b and c, if the length of a is higher than c, then a is still more similar to b than c. See this link for more detailed : 
```
https://developers.google.com/machine-learning/clustering/similarity/check-your-understanding
```


#### How is LSTM WORK ?

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to effectively capture long-term dependencies in sequential data. The key idea behind LSTMs is the use of a cell state that runs through the entire sequence, with various gates controlling the flow of information into and out of the cell state.

Here's a high-level overview of how an LSTM cell works:
- Forget Gate: Decides what information to discard from the cell state.
- Input Gate: Modifies the cell state by adding new information.
- Update: Calculate the new cell state by combining the results of the forget and input gates.
- Output Gate: Determines the output based on the updated cell state.
The design of LSTM allows it to better handle vanishing or exploding gradients, which are common issues in training traditional RNNs on long sequences.

Some common applications of LSTM include:
- Natural Language Processing (NLP): LSTMs are widely used in tasks such as language modeling, sentiment analysis, machine translation, and text generation.
- Speech Recognition: LSTMs are used in speech recognition systems to convert spoken language into text.
- Time Series Forecasting: LSTMs are effective for predicting future values based on historical data, making them useful in areas like stock market forecasting, weather prediction, and energy demand forecasting.
- Gesture Recognition: LSTMs can be applied to gesture recognition tasks, such as interpreting hand movements or body gestures.
- Healthcare: LSTMs are used in healthcare for tasks like predicting patient outcomes, analyzing medical records, and monitoring patient health data.


#### What is attention layer in NLP ?
```
https://www.geeksforgeeks.org/self-attention-in-nlp/
```


#### Difference between Masked Language Models (MLM) and Causal Language Models (CLM) in NLP
!.Masked Language Models (MLM):
- Objective: Predict the original masked words in a sentence.
- Training Data: Input sentences with random words masked out.
- Bidirectional Context: MLM can consider the entire context of a sentence because it predicts missing words based on both left and right contexts.
- Example Model: BERT (Bidirectional Encoder Representations from Transformers) is a popular MLM model.

2. Causal Language Models (CLM):
- Objective: Predict the next word in a sequence.
- Training Data: Input sequences where each word is predicted based only on the previous words.
- Unidirectional Context: CLM can only consider the context to the left of a word since it predicts the next word in a sequence.
- Example Model: GPT (Generative Pre-trained Transformer) is a well-known CLM model.

In summary, the main difference lies in how MLM and CLM are trained and the context they consider during prediction. MLM predicts masked words bidirectionally within a sentence, while CLM predicts the next word unidirectionally based on preceding words.

#### Could you explain what is transformer ?

The Transformer is a type of model architecture. The key innovation of the Transformer is the self-attention mechanism, which lets the model consider other words in the input sequence when processing each word.

Here's a simple overview of the Transformer architecture:
- Input Embedding: The input words are converted into vectors.
- Positional Encoding: Information about the position of each word in the input sequence is added to the embeddings.
- Encoder: Each encoder consists of two parts: a self-attention layer and a feed-forward neural network(*). The encoders are stacked on top of each other (the number varies based on the specific implementation).
- Decoder: Similar to the encoder, but with an additional layer of encoder-decoder attention.

(*)
In summary, the self-attention layer focuses on capturing relationships and dependencies between words in a sequence, allowing the model to weigh different words dynamically. On the other hand, the feedforward layer is responsible for introducing non-linearity and capturing complex patterns in the learned representations. Both layers play crucial roles in the overall functioning of transformer models and contribute to their effectiveness in various natural language processing tasks.

There are many model architecture based on Transformer such as:
- Bert : BERT, however, is bidirectional — it considers the context from both sides (left and right of a word) to understand the semantic meaning of a particular word in the sentence. Since BERT is trained bidirectionally, it is well suited for tasks that require understanding the context of both sides such as Question Answering, Named Entity Recognition, etc.
- RoBERTa (Robustly Optimized BERT Pretraining Approach): RoBERTa is a variation of BERT that uses dynamic masking rather than static masking(*) and trains on larger batches and for more steps than BERT.
- GPT (Generative Pre-training Transformer): GPT is a type of transformer model that is trained to predict the next word in a sentence (language modeling). It's unidirectional, meaning it only uses previous context (words to the left of the current word) for predictions.Since it's unidirectional,it is good at generating human-like text and it works better for tasks like text generation, chatbots, writing assistance etc.Even GPT work best for tasks tasks like text generation, it still can work for task such as question answering, for example GPT 3 of OpenAI.

(*) Dynamic Masking vs Static Masking: In Bert, the masking of tokens is static which means the model sees and predicts the masked tokens in the same positions during pre-training. RoBERTa uses dynamic masking: each time a sentence is fed into the model during training, the model selects different words to mask.