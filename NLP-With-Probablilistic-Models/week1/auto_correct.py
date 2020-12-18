"""
implement auto correct

1) identify misspelled word
    get a corpus dictionary
    if a word is not in dict, flag for correction

2) find strings n-edit distance away
    start with number in the range 1-3
    this is a hyperparameter
    find all possible strings that are n-edit distance

3) filter candidates
    only consider candidates that are in the dictionary
    if not in dict, remove

4) calculate word probabilities
    find most likely word from candidates
    most common - simple
    most likely given context - more complicated

    P(w) = C(w) / V
"""
from nltk.corpus import brown
from collections import Counter
import operator
import string
import numpy as np

class AutoCorrector():

    def __init__(self, max_dist=2):
        self.vocab = None
        self.candidates = None
        self.probabilites = None
        self.word = None
        self.max_dist = max_dist

    def create_vocab(self):
        self.vocab = set([w.lower() for w in brown.words()])

    def is_misspelled(self, word):
        return word not in self.vocab

    def calculate_probabilities(self):
        words = [w.lower() for w in brown.words()]
        word_counts = Counter(words)
        self.probabilites = {w:word_counts.get(w) / len(words) for w in word_counts.keys() }

    def _create_candidates(self, word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        inserts = [a[1:] + c + b for a, b in
                   map(lambda e: ('@' + e[0], e[1]), splits) for c in string.ascii_lowercase]
        deletes = [a[1:] + b[1:] for a, b in
                   map(lambda e: ('@' + e[0], e[1]), splits) if b]
        return set(inserts + deletes)

    def create_candidates(self, word):
        self.word = word
        self.candidates = set([word])
        for i in range(self.max_dist):
            for c in self.candidates:
                self.candidates = self.candidates.union(self._create_candidates(c))

    def filter_candidates(self) -> set:
        filtered =  set([w for w in self.candidates if w in self.vocab])
        self.candidates = filtered

    def min_edit_distance(self, a:str, b:str) -> int:
        # Creating numpy ndarray( initialized with 0 of dimension of size of both strings
        matrix = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int)

        # fill the respective index of matrix (row,column)
        for i in range(len(a) + 1):
            for j in range(len(b) + 1):

                # First doing the boundary value analysis
                # if first or second string is empty so directly adding insertion cost
                if i == 0:
                    matrix[i][j] = j
                    # Second case
                elif j == 0:
                    matrix[i][j] = i
                else:
                    matrix[i][j] = min(matrix[i][j - 1] + 1,
                                       matrix[i - 1][j] + 1,
                                       matrix[i - 1][j - 1] + 2 if a[i - 1] != b[j - 1] else matrix[i - 1][j - 1] + 0)
                    # Adjusted the cost accordinly, insertion = 1, deletion=1 and substitution=2
        return matrix[len(a)][len(b)]  # Returning the final

    def get_top_candidate(self) -> float:
        probs = {w:self.probabilites[w] for w in self.candidates}
        if probs == {}:
            return self.word
        return max(probs.items(), key=operator.itemgetter(1))[0]