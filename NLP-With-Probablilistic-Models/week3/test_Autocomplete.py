from unittest import TestCase
from Autocomplete import Autocomplete


class TestAutocomplete(TestCase):
    def setUp(self) -> None:
        self.ac = Autocomplete(3)
        self.ac0 = Autocomplete(0)
        self.sentence = "Here is my sentence!"

    def tearDown(self) -> None:
        pass

    def test_preprocess_sentence(self):
        processed1 = self.ac.preprocess_sentence(self.sentence)
        processed2 = self.ac0.preprocess_sentence(self.sentence)
        should_be1 = "<s> <s> here is my sentence </s>"
        should_be2 = "<s> here is my sentence </s>"
        self.assertEqual(should_be1, processed1,
                         "Error in sentence preprocessing")
        self.assertEqual(should_be2, processed2,
                         "Error in sentence preprocessing")

    def test__create_vocab(self, ""):
        text = "This is some toy data. Here is more data. The data are here."



    def test__create_count_matrix(self):
        pass