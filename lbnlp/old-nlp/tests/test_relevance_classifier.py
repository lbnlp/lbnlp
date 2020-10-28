from matscholar_core.nlp.relevance import RelevanceClassifier
import unittest

class RelevanceClassifierTest(unittest.TestCase):

    TEST_DOCS = ["The polymer may be used in OLEDs and biosensors.",
                 "The band gap of ZnO is 3.3 eV"]

    def test_clf(self):
        clf = RelevanceClassifier()
        preds0 = clf.classify(self.TEST_DOCS[0])
        preds1 = clf.classify(self.TEST_DOCS[1])
        self.assertEqual(preds0, 0)
        self.assertEqual(preds1, 1)

    def test_many(self):
        clf = RelevanceClassifier()
        preds = clf.classify_many(self.TEST_DOCS)
        self.assertEqual(preds[0], 0)
        self.assertEqual(preds[1], 1)
