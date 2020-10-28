import os

from monty.dev import requires

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import load_pickle, check_versions, model_loader_setup
from lbnlp.nlp.relevance import RelevanceClassifier
from lbnlp.nlp.process import MatScholarProcess

pkg = ModelPkgLoader("matscholar_2020v1")


@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):
    models_basepath = os.path.join(pkg.structured_path, "models")

    if model_name == "relevance_model":
        return load_relevance_model(models_basepath)

    elif model_name == "ner":
        return


def load_relevance_model(basepath):
    clf_path = os.path.join(basepath, f"relevance_model.p")
    tfidf_path = os.path.join(basepath, f"tfidf.p")
    processor = MatScholarProcess(phraser_path=os.path.join(basepath, "embeddings/phraser.pkl"))
    return RelevanceClassifier(clf_path, tfidf_path, processor)



if __name__ == "__main__":
    model = load("relevance_model", ignore_requirements=True)
    print(model.classify_many(["The polymer was used for an OLED. This can also be used for a biosensor.", "The bandgap of ZnO is 33 eV"]))