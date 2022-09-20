"""
Using word2vec to compare 2 sentences
"""

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.models import Word2Vec


# dataset = api.load("text8")
# model = Word2Vec(dataset)

# model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
# model.save("word2vec.model")

from gensim.test.utils import datapath
from gensim import utils

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

sentences = MyCorpus()
model = Word2Vec(sentences=sentences)

vec_king = model.wv['king']