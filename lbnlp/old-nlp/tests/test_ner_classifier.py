# from matscholar_core.nlp.ner import NERClassifier
# import unittest
#
# class NERClassifierTest(unittest.TestCase):
#
#     TEST_DOCS = ["We synthesized AO2 (A = Sr, Ba) thin films. The band gap was 2.5 eV.",
#             "The lattice constant of ZnO is 3.8 A. This was measured using XRD."]
#
#     def test_iob(self):
#         self.clf = NERClassifier(build=False)
#         tagged_docs =self.clf.as_iob(self.TEST_DOCS)
#         self.assertEqual(len(tagged_docs), 2)
#         self.assertEqual(len(tagged_docs[0]), 2)
#         self.assertEqual(tagged_docs[0][0][2][1], "B-MAT")
#
#
#     def test_concatenated(self):
#         self.clf = NERClassifier(build=False)
#         tagged_docs = self.clf.as_concatenated(self.TEST_DOCS)
#         self.assertEqual(len(tagged_docs), 2)
#         self.assertEqual(len(tagged_docs[0]), 2)
#         self.assertEqual(tagged_docs[0][0][2][1], "MAT")
#         self.assertEqual(tagged_docs[0][0][2][0], "AO2 ( A = Sr , Ba )")
#         self.assertFalse(any("-" in tag for token, tag in tagged_docs[0][0]))
#         self.clf.close_session()
#
#     def test_normalized(self):
#         self.clf = NERClassifier(build=False)
#         tagged_docs = self.clf.as_normalized(self.TEST_DOCS)
#         self.assertEqual(len(tagged_docs), 2)
#         self.assertEqual(len(tagged_docs[0]), 2)
#         self.assertEqual(tagged_docs[0][0][2][1], "MAT")
#         self.assertTrue(isinstance(tagged_docs[0][0][2][0], list))
#         self.clf.close_session()
#
# if __name__ == "__main__":
#     ner = NERClassifierTest()
#     ner.test_concatenated()
