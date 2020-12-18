from unittest import TestCase
from auto_correct import AutoCorrector
import time
import sys
sys.setrecursionlimit(2000)

class TestAutoCorrector(TestCase):
    def test_create_vocab(self):
        ac = AutoCorrector()
        ac.create_vocab()
        self.assertTrue(type(ac.vocab) == set)
        self.assertGreater(len(ac.vocab), 1000)

    def test_calculate_probabilities(self):
        ac = AutoCorrector()
        ac.create_vocab()
        ac.calculate_probabilities()

    def test_filter_candidates(self):
        ac = AutoCorrector()
        ac.create_vocab()
        candidates=['the', 'and', 'xdsdfggpppp']
        self.assertEqual(len(ac.filter_candidates(candidates)), 2)

    def test_min_edit_distance(self):
        ac = AutoCorrector()
        a = "hello"
        b = "hellp"
        c = "help"
        self.assertEqual(ac.min_edit_distance(a,b), 2)
        self.assertEqual(ac.min_edit_distance(a,c), 3)
        self.assertEqual(ac.min_edit_distance(b,c), 1)

    def test_create_candidates(self):
        ac = AutoCorrector()
        ac.create_vocab()
        word = "hello"
        ac.create_candidates(word)
        self.assertEqual(len(ac.candidates), 13961)
        self.assertIn('hellwlo', ac.candidates)

    def test_filter_candidates(self):
        ac = AutoCorrector()
        ac.create_vocab()
        word = "hello"
        ac.create_candidates(word)
        prev_len = len(ac.candidates)
        ac.filter_candidates()
        self.assertLessEqual(len(ac.candidates), prev_len)

    def test_get_top_candidate(self):
        start = time.time()
        ac = AutoCorrector()
        ac.create_vocab()
        ac.calculate_probabilities()
        words = ["hellpy", "threfore", "als", "xggf"]
        for word in words:
            ac.create_candidates(word)
            ac.filter_candidates()
            print(f"Original: {word}\nCorrected: {ac.get_top_candidate()}")
        end = time.time()
        print(f"Elapsed time: {end - start}")
