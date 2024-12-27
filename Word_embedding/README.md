---
word representation

---
# Overview of word representation

The picture below is the sumary of  word representation in NLP

![overview of word embedding](pictures/NLP_word_representation.png)

# Details of words representation

## Feature extractors

We use this approaches when we have a simple task which not involve **semantics** and **context**

Suppose that we have these example document:

* **Document 1** : "The cat sat on the mat."
* **Document 2** : "The dog barked."

### One hot encoding

Encodes individual words or characters as binary vectors.

Vocabulary:

`["the", "cat", "sat", "on", "mat", "dog", "barked"]`

Each word is represented as a binary vector with one "1" at the word's position in the vocabulary.

* **"cat"** → `[0, 1, 0, 0, 0, 0, 0]`
* **"dog"** → `[0, 0, 0, 0, 0, 1, 0]`

When to use:

* For simple tasks like basic NLP preprocessing.
* As input to machine learning models when relationships between words aren't needed.

### Bag of Words (BoW)

Represents documents based on word frequencies in the vocabulary.

* **Document 1** : `[2, 1, 1, 1, 1, 0, 0]`. "the" appears 2 times. "cat", "sat", "on", "mat" appear 1 time each.
* **Document 2** : `[1, 0, 0, 0, 0, 1, 1]`. "the" appears 1 time, "dog" and "barked" appear 1 time each.

When to use :

* For text classification or clustering where word frequency is important.
* For tasks with a small vocabulary or corpus size.

### TF-IDF

Represents documents by weighting word importance (frequency × rarity). The intuition behind this measure, is that a term (word) is very important if it appears many times inside a document AND the number of documents that the this term is present, is relatively small.

* **TF (Term Frequency):** Highlights terms frequently appearing in a specific document.
* **IDF (Inverse Document Frequency):** Reduces the importance of terms common across all documents (e.g., "the", "and").

When to use:

* For information retrivial, input is a query and the target is to check which document is relevant
* For text classificaiton, convert text to TF-IDF Vectors, then use the TF-IDF vectors as input features to a classification algorithm (e.g., Logistic Regression, SVM, Random Forest).
* **TF-IDF** effectively reduces the noise from common words, making it more useful when dealing with larger corpora. While BoW can create overly sparse and uninformative features in large vocabularies.
* Use TF-IDF when your dataset contains many high-frequency but non-informative words, the importance of these words is neutralized by TF-IDF, but it is escalated by BoW.
