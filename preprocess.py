# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:50:55 2018

@author: ATheb
"""

import gensim
#import scipy.sparse.linalg
# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  