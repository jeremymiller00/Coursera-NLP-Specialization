"""
A class to represent a trainable POS tagger
Using Hidden Markov Model
And the Viterbi Algorithm to find the most likely sequence of tags
"""
import numpy as np
from collections import Counter
from loguru import logger as log
log.add("logfile.log", level='DEBUG')

class Tagger():

    def __init__(self):
        self.transition_matrix = None
        self.emission_matrix = None
        self.c_matrix = None
        self.d_matrix = None
        self.tag_counts = None
        self.tag_pair_counts = None
        self.ordered_tag_pair_set = None
        self.ordered_tag_set = None
        self.vocab = None
        self.ordered_vocab = None
        self.N = None
        log.debug("Tagger created")

    def populate_transition_matrix(self, data, epsilon=0.0001):
        """
        Load the brown tagged sentence corpus from NLTK
        Define 'INIT' as the start of a sentence tag, transition to first tag
        :param data: nltk brown tagged sentence corpus, or similar
        :param epsilon: regularization; avoids p==1 and div by 0
        :return: None; populates transition matrix attribute
        """
        tag_pairs = []
        tag_counts = []
        log.info("Populate Transition Matrix Called")
        n_sent = 0
        for sent in data:
            tag_pairs.append(("INIT", sent[0][1]))
            # tag_counts.append("INIT")
            for i in range(len(sent) - 1):
                tag1 = sent[i][1]
                tag2 = sent[i + 1][1]
                tag_pairs.append((tag1, tag2))
                tag_counts.append(tag1)
            n_sent += 1
        log.debug("Tag pairs extracted")
        self.ordered_tag_set = sorted(set(tag_counts))
        self.N = len(self.ordered_tag_set)

        self.tag_counts = Counter(tag_counts)
        self.tag_pair_counts = Counter(tag_pairs)
        self.ordered_tag_pair_set = sorted(self.tag_pair_counts.keys())
        self.transition_matrix = np.zeros(
            (len(self.ordered_tag_set)+1, len(self.ordered_tag_set)),
            dtype="float32")
        log.debug("Transition matrix constructed")

        # create first row of transition matrix, init row
        for j in range(len(self.ordered_tag_set)):
            pair = ("INIT", self.ordered_tag_set[j])
            if pair in self.tag_pair_counts.keys():
                n = self.tag_pair_counts.get(pair) + epsilon
            else:
                n = epsilon
            d = n_sent + (self.N * epsilon)
            self.transition_matrix[0][j] = n / d

        # iterate through rest of tags pairs
        for i in range(len(self.ordered_tag_set)):
            for j in range(len(self.ordered_tag_set)):
                pair = (self.ordered_tag_set[i], self.ordered_tag_set[j])
                try:
                    n = self.tag_pair_counts.get(pair) + epsilon
                    d = self.tag_counts.get(self.ordered_tag_set[i]) + (self.N*epsilon)
                    self.transition_matrix[i+1][j] = n / d
                finally:
                    continue
        log.info("Transition matrix populated")
        log.debug(f"Transition matrix shape: {self.transition_matrix.shape}")
        log.debug(f"Transition matrix first row: \n{self.transition_matrix[0]}")

    def populate_emission_matrix(self, data, epsilon=0.00000001):
        """
        get all state_word counter
        rows of emission matrix should correspond to rows of transition matrix
        columns correspond to words in vocab
        i.e. each is a POS tag

        :param data: data: nltk brown tagged sentence corpus, or similar
        :param epsilon: epsilon: regularization; avoids p==1 and div by 0
        :return: None; populates emission matrix attribute
        """
        log.info("Populate Emission Matrix Called")
        c = Counter()
        self.vocab = set()
        for sent in data:
            lowered = [(word[1], word[0].lower()) for word in sent] # flip to Tag, String
            words = [word[0].lower() for word in sent] # get just the words
            self.vocab.update(words)
            c.update(lowered)
        log.debug("Words extracted for emission matrix")
        self.ordered_vocab = sorted(self.vocab)
        emission_matrix = np.ones(
            (len(self.ordered_tag_set), len(self.ordered_vocab)),
            dtype="float32")
        self.emission_matrix = emission_matrix * epsilon
        log.debug("Emission matrix constructed")
        for i in range(len(self.ordered_tag_set)):
            for j in range(len(self.ordered_vocab)):
                pair = (self.ordered_tag_set[i], self.ordered_vocab[j])
                if pair in c.keys():
                    n = c.get(pair) + epsilon
                    d = self.tag_counts.get(self.ordered_tag_set[i]) + (self.N*epsilon)
                    value = n / d
                    self.emission_matrix[i][j] = value
        log.info("Emission matrix populated")
        log.debug(f"Emission matrix shape : {self.emission_matrix.shape}")
        log.debug(f"Emission matrix first row sum: "
                  f"\n{np.sum(self.emission_matrix[0])}")
        log.debug(f"Emission matrix sum: {np.sum(self.emission_matrix)}")

    def viterbi_init(self, word):
        vocab_index = self.ordered_vocab.index(word)
        for i in range(len(self.ordered_tag_set)):
            a = self.transition_matrix[0][i]
            b = self.emission_matrix[i][vocab_index]
            self.c_matrix[i][0] = a * b

    def viterbi_fwd_pass(self, words, epsilon=0.00000001):
        for i in range(self.c_matrix.shape[0]):
            for j in range(1, self.c_matrix.shape[1]):
                k_value = self.d_matrix[i][j-1]
                p = 0.0
                vocab_index = self.ordered_vocab.index(words[j])
                for k in range(self.c_matrix.shape[0]):
                    a = self.c_matrix[k][j-1]
                    b = self.transition_matrix[k+1][i]
                    c = self.emission_matrix[i][vocab_index]
                    value = a * b * c
                    if value > p:
                        p = value
                        k_value = k
                self.c_matrix[i][j] = p
                self.d_matrix[i][j] = k_value

    def viterbi_bkwd_pass(self, words):
        tags = []
        # s is the row of D, also the tag index
        j = len(words) - 1
        last_col = self.c_matrix[:, j]
        s = np.argmax(last_col)
        while j >= 0:
            tags.append(self.ordered_tag_set[s])
            s = self.d_matrix[j][s]
            j -= 1
        tags.reverse()
        return list(zip(words, tags))

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

    def tag_sentence(self, sent):
        words = [w.lower() for w in sent.split()]
        self.c_matrix = np.zeros(
            (len(self.ordered_tag_set)-1, len(words)),
            dtype="float32")
        self.d_matrix = np.zeros(
            (len(self.ordered_tag_set)-1, len(words)),
            dtype="int32")
        self.viterbi_init(words[0])
        self.viterbi_fwd_pass(words)
        result = self.viterbi_bkwd_pass(words)
        return result

    def tag_text(self):
        pass
