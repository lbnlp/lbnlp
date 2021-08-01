import os

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import model_loader_setup
from lbnlp.relevance import RelevanceClassifier
from lbnlp.ner.clf import NERClassifier
from lbnlp.normalize import Normalizer
from lbnlp.process.matscholar import MatScholarProcess

from matbert_ner.predict import predict

pkg = ModelPkgLoader("matscholar_2020v1")


@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):
    models_basepath = os.path.join(pkg.structured_path, "models")

    if model_name == "aunp2":
        return load_aunp2_model(models_basepath)

    # elif model_name == "ner":
    #     return load_ner_model(models_basepath)
    #
    # elif model_name == "ner_simple":
    #     return load_ner_simple_model(models_basepath)





def load_aunp2_model(basepath):



