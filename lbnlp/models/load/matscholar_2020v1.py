import os

from monty.dev import requires

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import load_pickle, check_versions
from lbnlp.nlp.relevance import RelevanceClassifier

pkg = ModelPkgLoader("matscholar_2020v1")


def load(model_name, ignore_requirements=False):
    pkg.load()
    if model_name not in pkg.model_names:
        raise ValueError(f"Model {model_name} in {pkg.modelpkg_name} not found. Choose from {pkg.model_names}")
    models_basepath =  os.path.join(pkg.structured_path, "models")
    if not ignore_requirements:
        check_versions(pkg.model_requirements[model_name])

    if model_name == "relevance_model":
        return load_relevance_model(models_basepath)

    elif model_name == "ner":
        return


def load_relevance_model(basepath):
    clf_path = os.path.join(basepath, f"relevance_model.p")
    tfidf_path = os.path.join(basepath, f"tfidf.p")
    return RelevanceClassifier(clf_path, tfidf_path)



if __name__ == "__main__":
    model = load("relevance_model", ignore_requirements=True)
    print(model.classify_many([
                                "The polymer was used for an OLED. This can also be sued for a biosensor.",
                                "The bandgap of ZnO is 33 eV"]))