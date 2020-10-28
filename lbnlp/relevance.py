import dill
import numpy as np

from lbnlp.process.matscholar import MatScholarProcess


class RelevanceClassifier:
    """
    A class to classify documents as relevant/not-relevant to inorganic materials science
    """

    def __init__(self, clf_path, tfidf_path, processor=None):
        """
        Constructor method for RelevanceClassifier. Loads the classifier and tfidf transformer.
        """

        self.processor = processor if processor else MatScholarProcess()
        with open(clf_path, "rb") as f:
            self.clf = dill.load(f)
        with open(tfidf_path, "rb") as f:
            self.tfidf = dill.load(f)

    def _preprocess(self, text):
        """
        Performs pre-processing (tokenization, lowering, etc).

        :param text: string; document to be processed
        :return: array; the processed tokens
        """

        sents = self.processor.tokenize(text)
        processed_sents = []
        for sent in sents:
            processed, _ = self.processor.process(sent)
            processed_sents.append(processed)

        flattened = [token for sent in processed_sents for token in sent]
        return flattened


    def classify(self, doc, decision_boundary=0.5):
        """
        Classify a document as relevant or not relevant

        :param doc: string; a document to be classified
        :param decision_boundary: float; probability required for a positive classification
        :return: int; either 1 or 0 (relevant or not relevant)
        """

        processed = self._preprocess(doc)
        X = self.tfidf.transform([processed])
        prob = self.clf.predict_proba(X)[0][1]
        pred = 1 if prob >= decision_boundary else 0
        return pred

    def classify_many(self, docs, decision_boundary=0.5):
        """
        Classify multiple documents as relevant or not relevant

        :param docs: list; a list of documents (as a string) to be classified
        :param decision_boundary: float; probability required for a positive classification
        :return: array; predicted labels (1 or 0)
        """

        processed = [self._preprocess(doc) for doc in docs]
        X = self.tfidf.transform(processed)
        prob = self.clf.predict_proba(X)[:, 1]
        preds = np.where(prob > decision_boundary, 1, 0)
        return preds


if __name__ == "__main__":
    clf = RelevanceClassifier()
    print(clf.classify_many(["The polymer was used for an OLED. This can also be sued for a biosensor.", "The bandgap of ZnO is 33 eV"]))









