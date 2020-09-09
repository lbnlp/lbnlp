import unittest

from matscholar_core.processing.word2vec import DataPreparation


class Word2VecProcessingTest(unittest.TestCase):

    def setUp(self):

        self.dp = DataPreparation()

        # lowercase, material normalization, converting to <nUm>
        self.test_sentence_1 = "We measured 100 materials , including Ni(CO)4 and obtained " \
                               "very high Thermoelectric Figures of merit ZT .".split()
        self.test_processed_sentence_1 = "we measured <nUm> materials , including C4NiO4 and obtained very " \
                                         "high thermoelectric figures of merit ZT .".split()
        self.test_processed_sentence_1_nopunkt = "we measured <nUm> materials including C4NiO4 and obtained very " \
                                                 "high thermoelectric figures of merit ZT".split()
        self.test_processed_sentence_1_nonum = "we measured 100 materials , including C4NiO4 and obtained very " \
                                               "high thermoelectric figures of merit ZT .".split()
        self.test_processed_sentence_1_nomatnorm = "we measured <nUm> materials , including Ni(CO)4 and obtained very " \
                                                   "high thermoelectric figures of merit ZT .".split()
        self.test_split_indices_1 = []

        # lowercase, splitting units and splitting valence state, convert to <nUm>
        self.test_sentence_2 = "Fe(II) was oxidized to obtain 5mol. Ferrous Oxide".split()
        self.test_processed_sentence_2 = "Fe (II) was oxidized to obtain <nUm> mol. ferrous oxide".split()
        self.test_processed_sentence_2_nonum = "Fe (II) was oxidized to obtain 5 mol. ferrous oxide".split()
        self.test_split_indices_2 = [0, 5]  # indices of tokens that were split

    def test_process_sentence(self):

        # sentence 1
        processed_sentence_1, split_indices_1 = self.dp.process_sentence(self.test_sentence_1)
        self.assertListEqual(processed_sentence_1, self.test_processed_sentence_1)
        self.assertListEqual(split_indices_1, self.test_split_indices_1)

        # sentence 1, no punctuation
        processed_sentence_1, split_indices_1 = self.dp.process_sentence(self.test_sentence_1, exclude_punct=True)
        self.assertListEqual(processed_sentence_1, self.test_processed_sentence_1_nopunkt)

        # sentence 1, no <nUm> conversion
        processed_sentence_1, split_indices_1 = self.dp.process_sentence(self.test_sentence_1, convert_num=False)
        self.assertListEqual(processed_sentence_1, self.test_processed_sentence_1_nonum)

        # sentence 1, no material normalization
        processed_sentence_1, split_indices_1 = self.dp.process_sentence(self.test_sentence_1, normalize_mats=False)
        self.assertListEqual(processed_sentence_1, self.test_processed_sentence_1_nomatnorm)

        # sentence 2
        processed_sentence_2, split_indices_2 = self.dp.process_sentence(self.test_sentence_2)
        self.assertListEqual(processed_sentence_2, self.test_processed_sentence_2)
        self.assertListEqual(split_indices_2, self.test_split_indices_2)

        # sentence 2, no <nUm> conversion
        processed_sentence_2, split_indices_2 = self.dp.process_sentence(self.test_sentence_2, convert_num=False)
        self.assertListEqual(processed_sentence_2, self.test_processed_sentence_2_nonum)
        self.assertListEqual(split_indices_2, self.test_split_indices_2)

    def test_is_number(self):

        self.assertTrue(self.dp.is_number("-5"))
        self.assertTrue(self.dp.is_number("-5.5"))
        self.assertTrue(self.dp.is_number("123.4"))
        self.assertTrue(self.dp.is_number("1,000,000"))

        self.assertFalse(self.dp.is_number("not a number"))
        self.assertFalse(self.dp.is_number("with23numbers"))
        self.assertFalse(self.dp.is_number("23.54b"))
        self.assertFalse(self.dp.is_number("-23a"))

    def test_is_simple_formula(self):

        self.assertTrue(self.dp.is_simple_formula("CoO"))
        self.assertTrue(self.dp.is_simple_formula("H2O"))
        self.assertTrue(self.dp.is_simple_formula("C(OH)2"))
        self.assertTrue(self.dp.is_simple_formula("C(OH)2Si"))
        self.assertTrue(self.dp.is_simple_formula("Ni0.5(CO)2"))

        self.assertFalse(self.dp.is_simple_formula("Ad2Al"))
        self.assertFalse(self.dp.is_simple_formula("123"))
        self.assertFalse(self.dp.is_simple_formula("some other text"))
        self.assertFalse(self.dp.is_simple_formula("Ni0.5(CO)2)"))

    def test_get_ordered_integer_formula(self):

        self.assertEqual(self.dp.get_ordered_integer_formula({"Al": 1, "O": 1.5}), "Al2O3")
        self.assertEqual(self.dp.get_ordered_integer_formula({"C": 0.04, "Al": 0.02, "O": 0.5}), "AlC2O25")

    def test_get_norm_formula(self):

        self.assertEqual(self.dp.get_norm_formula("SiO2"), "O2Si")
        self.assertEqual(self.dp.get_norm_formula("Ni(CO)4"), "C4NiO4")
        self.assertEqual(self.dp.get_norm_formula("Ni(CO)4SiO2"), "C4NiO6Si")
