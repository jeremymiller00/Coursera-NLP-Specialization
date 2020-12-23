"""
A class to represent a trainable POS tagger
Using Hidden Markov Model
And the Viterbi Algorithm to find the most likely sequence of tags
"""
import numpy as np
from collections import Counter


class Tagger():

    def __init__(self):
        self.transition_matrix = None
        self.emission_matrix = None
        self.c_matrix = None
        self.d_matrix = None
        self.tag_counts = None
        self.tag_pair_counts = None

    def populate_transition_matrix(self, data, epsilon=0.0001):
        """
        Load the brown tagged sentence corpus from NLTK
        :param data: nltk brown tagged sentence corpus, or similar
        :param epsilon: regularization; avoids p==1 and div by 0
        :return: None; populates transition matrix attribute
        """
        tag_pairs = []
        tag_counts = []
        for sent in data:
            for i in range(len(sent) - 1):
                tag1 = sent[i][1]
                tag2 = sent[i + 1][1]
                tag_pairs.append((tag1, tag2))
                tag_counts.append(tag1)
        N = len(set(tag_counts))
        self.tag_counts = Counter(tag_counts)
        self.tag_pair_counts = Counter(tag_pairs)
        self.transition_matrix = {}

        for tag_pair in self.tag_pair_counts.keys():
            numerator = self.tag_pair_counts.get(tag_pair) + epsilon
            denominator = self.tag_counts.get(tag_pair[0]) + (N*epsilon)
            self.transition_matrix[tag_pair] = numerator / denominator

    def populate_emission_matrix(self, data):
        # get all state_word counter
        c = Counter()
        for sent in data:
            lowered = [(word[1], word[0].lower()) for word in sent] # flip to Tag, String
            c.update(lowered)
        self.emission_matrix = c


    def viterbi_init(self):
        pass

    def viterbi_fwd_pass(self):
        pass

    def viterbi_bkwd_pass(self):
        pass

    def load_training_data(self):
        pass

    def split_data(self):
        pass

    def fit(self, data):
        self.populate_transition_matrix(data)
        self.populate_emission_matrix(data)

    def save_model(self):
        pass

    def load_model(self):
        pass

    def tag_sentence(self):
        pass

    def tag_text(self):
        pass
