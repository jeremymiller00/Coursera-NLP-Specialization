import numpy as np
import string
from loguru import logger as log


class Autocomplete():

    def __init__(self, N: int):
        self.N = N
        self.count_matrix = None
        self.probability_matrix = None
        self.vocab = None
        log.debug("Autocompleter created")

    def preprocess_sentence(self, sentence: str):
        # prepend with n-1 start sentence token
        # append with 1 end sentence token
        cleaned = "".join([ch.lower() for ch in sentence if ch not in string.punctuation])
        if self.N > 1:
            start_tags = self.N - 1
        else:
            start_tags = 1
        return (start_tags * "<s> ") + cleaned + " </s>"

    def fit(self, path_to_corpus: str):
        self._create_vocab(path_to_corpus)
        self._create_count_matrix()
        self._create_probability_matrix()
        pass

    def autocomplete(self, text: str) -> str:
        """

        :param text:
        :return:
        """
        pass

    def generate_text(self, starter: str, k: int) -> str:
        """
        Generate sentences from starter text
        :param starter:
        :param k:
        :return:
        """
        pass

    def _create_vocab(self, path_to_corpus):
        pass

    def _create_count_matrix(self):
        # need a vocab, include start and end tokens
        # need a vocab of bigrams
        # each bigram is a row in the matrix
        # col value is count where token follows bigram

        pass

    def _create_probability_matrix(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass


