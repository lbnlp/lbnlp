
import os

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import model_loader_setup
from lbnlp.relevance import RelevanceClassifier
from lbnlp.process.matscholar import MatScholarProcess

pkg = ModelPkgLoader("relevance_2020v1")


@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):
    models_basepath = os.path.join(pkg.structured_path, "relevance_2020v1 copy/models")

    if model_name == "relevance":
        return load_relevance_model(models_basepath)


def load_relevance_model(basepath):
    clf_path = os.path.join(basepath, f"relevance_model.p")
    tfidf_path = os.path.join(basepath, f"tfidf.p")
    processor = MatScholarProcess(phraser_path=os.path.join(basepath, "embeddings/phraser.pkl"))
    return RelevanceClassifier(clf_path, tfidf_path, processor)