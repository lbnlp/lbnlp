import os

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import load_pickle, check_versions, model_loader_setup
from lbnlp.relevance import RelevanceClassifier
from lbnlp.ner.clf import NERClassifier
from lbnlp.normalize import Normalizer
from lbnlp.process.matscholar import MatScholarProcess

pkg = ModelPkgLoader("matscholar_2020v1")


@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):
    models_basepath = os.path.join(pkg.structured_path, "models")

    if model_name == "relevance_model":
        return load_relevance_model(models_basepath)

    elif model_name == "ner":
        return load_ner_model(models_basepath)


def load_relevance_model(basepath):
    clf_path = os.path.join(basepath, f"relevance_model.p")
    tfidf_path = os.path.join(basepath, f"tfidf.p")
    processor = MatScholarProcess(phraser_path=os.path.join(basepath, "embeddings/phraser.pkl"))
    return RelevanceClassifier(clf_path, tfidf_path, processor)


def load_ner_model(basepath):
    ner_path = os.path.join(basepath, "ner")

    processor = MatScholarProcess(phraser_path=os.path.join(basepath, "embeddings/phraser.pkl"))
    normalizer = Normalizer(os.path.join(basepath, "normalize"), os.path.join(basepath, "rsc"))
    return NERClassifier(ner_path, normalizer, processor, enforce_local=True)


if __name__ == "__main__":
    # model = load("relevance_model")
    # print(model.classify_many(["The polymer was used for an OLED. This can also be used for a biosensor.", "The bandgap of ZnO is 33 eV"]))

    model = load("ner")


    doc = "CoCrPt/CoCr/carbon films were sputter-deposited on CoTaZr soft-magnetic underlayers and the effects of a carbon intermediate layer on magnetic and recording properties were investigated by changing a heating sequence in sample preparations. A heating process before a CoCr deposition was needed to obtain a high perpendicular coercivity. The carbon diffusion into a CoCr layer during its deposition led to small crystal grains in the CoCr layer and thereby the CoCrPt layer. Consequently, a high perpendicular coercivity was obtained, which was considered due to the change in magnetization process from a wall motion to a coherent rotation. The use of a thin (1–5nm) carbon intermediate layer was found to be effective to obtain both low noise and high resolution."
    tags = model.tag_doc(doc)

    print(tags)