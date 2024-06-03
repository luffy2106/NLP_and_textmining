# NLP Interview Questions
A collection of technical interview questions for NLP  positions.

Some popular github :
- https://github.com/khangich/machine-learning-interview



#### What is Word embedding

### Word Embedding

Word embedding is a technique used in natural language processing (NLP) to represent words as vectors of real numbers. These vectors capture the semantic meaning of words based on their usage in context. By representing words as dense vectors, word embedding techniques help machine learning models better understand and process text data.

There are various methods to create word embeddings, such as Word2Vec, GloVe, and FastText. These methods learn the vector representations of words by considering their relationships with other words in a large corpus of text data. The resulting word embeddings can then be used in tasks like sentiment analysis, text classification, machine translation, and more.

Word embeddings have become an essential component in NLP applications, enabling models to effectively capture the meaning of words and improve performance on various textual tasks.

#### Word embedding category 

There are several word embedding methods which can be divided into two major categories : 𝗖𝗼𝗻𝘁𝗲𝘅𝘁-𝗶𝗻𝗱𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁 and 𝗖𝗼𝗻𝘁𝗲𝘅𝘁-𝗱𝗲𝗽𝗲𝗻𝗱𝗲𝗻𝘁

1. Context-independent methods are characterized by being unique and distinct for each word without considering the word’s context.

🔹 Context-independent without machine learning

- 𝗕𝗮𝗴-𝗼𝗳-𝘄𝗼𝗿𝗱𝘀: This method represents a text, such as a sentence or a document, as the bag of its words, disregarding grammar and even word order but keeping multiplicity.

- 𝗧𝗙-𝗜𝗗𝗙: This gets this importance score by getting the term’s frequency (TF) and multiplying it by the term inverse document frequency (IDF).

🔹 Context-independent with machine learning

- 𝗪𝗼𝗿𝗱𝟮𝗩𝗲𝗰: A shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words, utilize either of two model architectures: continuous bag-of-words (CBOW) or continuous skip-gram.

- 𝗚𝗹𝗼𝗩𝗲: This performs training on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

- 𝗙𝗮𝘀𝘁𝗧𝗲𝘅𝘁: This method embeds words by treating each word as being composed of character n-grams instead of a word whole. This feature enables it not only to learn rare words but also out-of-vocabulary words.

2. Context-dependent learns different embeddings for the same word based on its context.

🔹 Context-dependent and RNN based

- 𝗘𝗟𝗠𝗢: learns contextualized word representations based on a neural language model with a character-based encoding layer and two BiLSTM layers.

- 𝗖𝗼𝗩𝗲: uses a deep LSTM encoder from an attentional sequence-to-sequence model trained for machine translation to contextualize word vectors.

🔹 Context-dependent and transformer-based

- 𝗕𝗘𝗥𝗧: This is a transformer-based language representation model trained on a large cross-domain corpus, which uses a masked language model to predict words that are randomly masked in a sequence.

- 𝗫𝗟𝗠: Another transformer based model which pretrained using next token prediction, masked language modeling and a translation objective.

- 𝗥𝗼𝗕𝗘𝗥𝗧𝗮: This is built on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.

- 𝗔𝗟𝗕𝗘𝗥𝗧: This is a parameter-reduction techniques to lower memory consumption and increase the training speed of BERT.

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
Considering a window size of 1 (i.e., for each word, you look to the 1 to the left and one to the right)

So when you input ‘chicken ‘ into the network, it will try to predict the closest word as ‘like’ or ‘wings’.

Thus it forms labels and trains the network.

Word2Vec is not a true unsupervised learning technique (since there is some sort of error backpropagation taking place through correct and incorrect predictions), they are a self-supervised technique, a specific instance of supervised learning where the targets are generated from the input data. In order to get self-supervised models to learn interesting features, you have to come up with an interesting synthetic target and loss function.

#### How to evaluate word2vec/doc2vec :

Word2Vec/doc2vec training is an unsupervised task, there’s no good way to objectively evaluate the result. Evaluation depends on your end application(human eyes)


#### What is seq2seq model 

use mainly for translation
https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

#### Explain transformer 

Transformer including input, encoders, decoders, output. The encoding components is a stack of encoders, The decoding component is a stack of decoders of the same number.

https://jalammar.github.io/illustrated-transformer/


#### Explain the difference between gensim and transformer 
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


#### How RNN work ?

The main difference between Feed Forward Neural Network and ss RNN is :

Traditional feed-forward neural networks take in a fixed amount of input data all at the same time and produce a fixed amount of output each time. On the other hand, RNNs do not consume all the input data at once. Instead, they take them in one at a time and in a sequence. At each step, the RNN does a series of calculations before producing an output. The output, known as the hidden state, is then combined with the next input in the sequence to produce another output. This process continues until the model is programmed to finish or the input sequence ends.

Reference :
- https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
- https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html


#### What is the transformer encoder decoder ?(need to research in detailed)
The transformer uses an encoder-decoder architecture. The encoder extracts features from an input sentence, and the decoder uses the features to produce an output sentence (translation).
- The encoder in the transformer consists of multiple encoder blocks. An input sentence goes through the encoder blocks, and the output of the last encoder block becomes the input features to the decoder.
- The decoder also consists of multiple decoder blocks.

https://kikaben.com/transformers-encoder-decoder/
Reference : http://cs231n.stanford.edu/slides/2021/lecture_11.pdf

#### How BERT work, why do you want to use BERT
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
#### Explain what is semantic analysis


Semantic analysis, also known as **semantic parsing** or **natural language understanding**, is the process of understanding the meaning of words, phrases, sentences, or even entire documents in natural language. This involves interpreting the context, relationships, and intended communication conveyed by the text.

Semantic analysis goes beyond syntax (grammatical structure) and focuses on extracting the underlying meaning and information from the text. It aims to comprehend the semantics of language to enable machines to understand human language in a meaningful way.

Key tasks involved in semantic analysis include:

1. **Named Entity Recognition (NER)**: Identifying and classifying entities mentioned in the text such as names of people, organizations, locations, dates, etc.

2. **Sentiment Analysis**: Determining the sentiment or opinion expressed in the text (positive, negative, neutral).

3. **Word Sense Disambiguation**: Resolving the meaning of words with multiple meanings based on the context in which they are used.

4. **Text Classification**: Categorizing text into predefined categories or labels based on its content.

5. **Relationship Extraction**: Identifying relationships between entities mentioned in the text.

6. **Question Answering**: Understanding questions posed in natural language and providing relevant answers.

Semantic analysis is essential in various applications such as search engines, chatbots, virtual assistants, and sentiment analysis tools to improve accuracy and effectiveness in processing and understanding human language.


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
Therefore, the anwser is 1 - LSTM

#### What are the most 3 relevant metrics to determine the similarity between 2 words/sentences with normalized embedding ?
1. Cosine
2. Jaccard
3. Euclidian
4. Hamming 
5. Dot Product

Top 3 Relevant Metrics for Similarity Measurement with Normalized Embeddings:

1. **Cosine Similarity:**
   - Cosine similarity measures the cosine of the angle between two vectors in a multi-dimensional space.
   - It is widely used for measuring similarity between normalized vectors, such as word embeddings.
   - The range of cosine similarity is [-1, 1], where 1 indicates identical vectors and -1 indicates completely opposite vectors.

2. **Euclidean Distance:**
   - Euclidean distance calculates the straight-line distance between two points in Euclidean space.
   - While not commonly used for measuring similarity directly, it can still provide insights into the dissimilarity between normalized vectors.
   - The smaller the Euclidean distance, the more similar the vectors are.

3. **Dot Product:**
   - The dot product measures the projection of one vector onto another.
   - When working with normalized embeddings, the dot product can also serve as a metric for similarity.
   - A higher dot product value indicates greater similarity between the vectors.

These three metrics are commonly utilized for determining similarity between words or sentences represented as normalized embeddings. Each metric offers unique insights into the relationship between vectors and can be chosen based on specific requirements or characteristics of the data being analyzed.


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
### **Attention Layer in Natural Language Processing (NLP) - Simplified Explanation:**

In NLP, an attention layer helps the neural network to pay more attention to important words when processing text. It works like focusing a flashlight on specific words while reading a sentence.

- **How it Works:**
  - Imagine you are reading a sentence, and some words are more crucial for understanding the meaning.
  - The attention layer helps the model to identify and focus on these important words during processing.

- **Key Points:**
  - **Query:** The current word being processed.
  - **Key:** Other words in the sentence.
  - **Value:** Importance of each word.

- **Benefits:**
  - Makes the model better at understanding complex sentences.
  - Improves performance in tasks like translation or summarization.
  - Helps the model to learn relationships between words effectively.

- **Types:**
  - **Self-Attention:** Comparing words within the same sentence.
  - **Multi-Head Attention:** Looking at different aspects of the text simultaneously.

Overall, the attention layer guides the model to focus on essential parts of the text, making it smarter and more accurate in processing language data.

Reference:
```
https://www.geeksforgeeks.org/self-attention-in-nlp/
```

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



#### What is the difference between Embedding model and Large Language Model ?

Primary Function:
- Embedding Model: Converts text into vector representations(for feature extractors)
- LLM: Understands and generates text, capable of handling a wide range of NLP tasks.
Output:
- Embedding Model: Produces fixed-size vectors.
- LLM: Produces text or performs specific NLP tasks
Applications:
- Embedding Model : 
* Similarity measurement (e.g., finding similar documents or words)
* Feature representation for downstream tasks (e.g., classification, clustering)
* Information retrieval and search.
- LLM model:
* Text generation (e.g., writing articles, generating code).
* Conversational agents and chatbots.
* Language translation.
* Text summarization.
* Sentiment analysis and more.

You can choose embedding models from the leaderboards, based on your use cases
```
https://huggingface.co/spaces/mteb/leaderboard
```
Example :
- Examples of Embedding Models:
* Word2Vec
* GloVe (Global Vectors for Word Representation)
* FastText
* BERT 
- Examples of Large Language Models (LLMs):
* GPT-3
* GPT-4
* BERT
* Roberta
* Mistral
* Llama 2,3










