import os

from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import model_loader_setup

from matbert_ner.predict import predict

pkg = ModelPkgLoader("matbert_ner_2021v1")


@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):
    models_basepath = os.path.join(pkg.structured_path, "matbert_ner_models")
    return MatBERTNERModelWrapper(model_name=model_name, basepath=models_basepath)


class MatBERTNERModelWrapper:
    """
    A wrapper around the core "predict" method of lbnlp/MatBERT-NER.

    Args:
        model_name (str): The name of the model to load.
        basepath (str): The base path to this model package's inner data files.

    Attributes:
        model_file (str): the absolute path to the base MatBERT pretrained model file.
        state_path_file (str): the absolute path to the fine-tuned MatBERT-NER model state.
    """

    def __init__(self, model_name, basepath):
        self.model_file = os.path.abspath(os.path.join(basepath, "model_files/matbert-base-uncased"))

        sp_basedir = os.path.join(basepath, "state_paths")
        if model_name == "aunp2":
            state_path_dir =  os.path.join(sp_basedir, "matbert_aunp2_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_80")
        elif model_name == "aunp11":
            state_path_dir = os.path.join(sp_basedir, "matbert_aunp11_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_80")
        elif model_name == "doping":
            state_path_dir = os.path.join(sp_basedir, "matbert_doping_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_80")
        elif model_name == "solid_state":
            state_path_dir = os.path.join(sp_basedir, "matbert_solid_state_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_80")
        else:
            raise NameError(f"No MatBERT-NER model is known as '{model_name}'.")

        self.state_path_file = os.path.abspath(os.path.join(state_path_dir, "best.pt"))

    def tag_docs(self, texts, device="cpu"):
        """
        Tag a list of documents (texts, as strings before tokenization) with MatBERT-NER using the NER model selected by __init__.

        Documents are internally tagged with the IOBES scheme and then are rejoined to output.

        Args:
            texts ([str]): List of documents to tag.
            device (str): Device to run inference with (e.g., "cpu", "gpu") as interpretable by PyTorch.

        Returns:

        """
        predictions = predict(texts, model_file=self.model_file, state_path=self.state_path_file, scheme="IOBES", device=device)
        return predictions

