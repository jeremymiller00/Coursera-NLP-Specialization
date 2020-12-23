"""
Load data and train a POS tagger
"""
from Tagger import Tagger
from nltk.corpus import brown
import numpy as np

def load_data(test_prop=0.9):
    """
    Load the Brown tagged training corpus from NLTK
    :param test_prop: proportion of corpus to use as test data
    :return: training data, test data
    """
    brown_tagged_sents = brown.tagged_sents(categories='news')
    div = int(len(brown_tagged_sents) * test_prop)
    # idx = np.array(list(range(len(brown_tagged_sents))), dtype='int32')
    # np.random.shuffle(idx)
    # train_idx = idx[:div]
    # test_idx = idx[div:]
    train = brown_tagged_sents[:div]
    test = brown_tagged_sents[div:]
    print(f"{len(train)} training sentences, {len(test)} test sentences")
    return train, test

def evaluate_tagger(tagger, test):
    pass

#####################################


if __name__ == '__main__':

    train, test = load_data()
    tagger = Tagger()
    tagger.fit(train)
    # evaluate_tagger(tagger, test)