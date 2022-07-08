import os

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import model_loader_setup
from lbnlp.ner.clf import NERClassifier
from lbnlp.normalize import Normalizer
from lbnlp.process.matscholar import MatScholarProcess

pkg = ModelPkgLoader("matscholar_2020v1")


@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):
    models_basepath = os.path.join(pkg.structured_path, "models")

    if model_name == "ner":
        return load_ner_model(models_basepath)

    elif model_name == "ner_simple":
        return load_ner_simple_model(models_basepath)


def load_ner_model(basepath):
    ner_path = os.path.join(basepath, "ner")

    processor = MatScholarProcess(phraser_path=os.path.join(basepath, "embeddings/phraser.pkl"))
    normalizer = Normalizer(os.path.join(basepath, "normalize"), os.path.join(basepath, "rsc"))
    return NERClassifier(ner_path, normalizer, processor, enforce_local=True)


def load_ner_simple_model(basepath):
    ner_path = os.path.join(basepath, "ner")
    return NERClassifierConvenienceWrapper(ner_path, basepath)


class NERClassifierConvenienceWrapper:
    """
    A convenience wrapper for the frozen NERClassifier, which does some convenient things:

     - multitoken entities, simplifying IOB2 scheme to simple entity scheme
     - PVT/PUT property simplification to PRO entities

    """
    def __init__(self, ner_path, basepath):
        self.processor = MatScholarProcess(phraser_path=os.path.join(basepath, "embeddings/phraser.pkl"))
        self.normalizer = Normalizer(os.path.join(basepath, "normalize"), os.path.join(basepath, "rsc"))
        self.clf = NERClassifier(ner_path, self.normalizer, self.processor, enforce_local=True)

    def tag_doc(self, doc):
        tagged = self.clf.tag_doc(doc)

        new_tagged = []

        property_remapped_b = ["B-PVL", "B-PUT"]
        property_remapped_i = ["I-PVL", "I-PUT"]

        for sentence in tagged:
            new_sentence = []
            for token, ent in sentence:
                if ent in property_remapped_b:

                    # In the case there is a B-PVL followed by a B-PUT
                    # it will be converted to B-PRO B-PRO instead of
                    # B-PRO I-PRO, so it must be converted if previous token
                    # Is also a property entity
                    if new_sentence:
                        if new_sentence[-1][1] in ("B-PRO", "I-PRO"):
                            new_ent = "I-PRO"
                        else:
                            new_ent = "B-PRO"
                    else:
                        new_ent = "B-PRO"

                elif ent in property_remapped_i:
                    new_ent = "I-PRO"
                else:
                    new_ent = ent
                new_sentence.append((token, new_ent))
            new_tagged.append(new_sentence)

        tagged_concat = self.normalizer._concatenate_ents(new_tagged)
        return tagged_concat

