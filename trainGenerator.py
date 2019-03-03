# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:59:00 2018

@author: ATheb
"""
import numpy as np
from tensorflow.keras.utils import Sequence
import gensim   
    
class DataGenerator(Sequence):
    def __init__(self, sentence_list, next_word_list, batch_size, maxlen, model):
        self.batch_size = batch_size
        self.sentence_list = sentence_list
        self.next_word_list = next_word_list
        self.maxlen = maxlen
        self.model = model

    def __len__(self):
        return int(len(self.sentence_list)/self.batch_size) + 1

    def __getitem__(self, idx):
        # batch size depends on number of words per sentence
        # can use different lengths for batches
        # inside batch has to have the same length ->sentence packages
        x = np.empty((self.batch_size, self.maxlen, 100), dtype=np.float64)
        y = np.empty((self.batch_size, 100), dtype=np.float64)
        
        for i in range(idx, idx+self.batch_size):
            for t in range(0, self.maxlen):
                word = self.sentence_list[i][t]
                if word in self.model.wv.vocab:
                    wordVec = self.model.wv[word]
                else:
                    wordVec = np.zeros(100)
                x[i-idx, t] = wordVec
            goal = self.next_word_list[i]
            if goal in self.model.wv.vocab:
                goalVec = self.model.wv[goal]
            else:
                goalVec = np.zeros(100)
            y[i-idx] = goalVec
        return x, y
