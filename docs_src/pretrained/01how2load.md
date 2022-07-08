# Loading pretrained models


LBNLP provides easy access to many ready-to-use models for NLP tasks such as:

- **Named entity recognition** for inorganic materials (via LSTM)
- **Relevance classification** for materials science-related texts (via Logistic Regression)
- **Suggestions for scientific journals** (via Doc2Vec)
- **Word2Vec embeddings** for various materials-related entities
- And more!



## View available models

View what models are currently available with `print_models_info`
```python
from lbnlp.models.rolodex import print_models_info

print_models_info()
```


Related models and tools are packaged together in "model packages".


```stdout
Model Package: 'matscholar_2020v1'
	* 'ner': Named entity recognition for materials.
		 - More info: https://doi.org/10.1021/acs.jcim.9b00470
	* 'ner_simple': A simpler tagging scheme version of the regular matscholar2020v1 NER model.
		 - More info: https://doi.org/10.1021/acs.jcim.9b00470
Model Package: 'relevance_2020v1'
	* 'relevance': Classifier for predicting relevance to materials science text
		 - More info: https://doi.org/10.1021/acs.jcim.9b00470
Model Package: 'matbert_ner_2021v1'
	* 'aunp11': Named entity recognition for gold nanoparticle with 11 labels using BERT pre-trained on materials science text.
		 - More info: https://doi.org/10.1016/j.patter.2022.100488
	* 'aunp2': Named entity recognition for gold nanoparticle adjective descriptions (DES) and noun morphologies (MOR) using BERT pre-trained on materials science text.
		 - More info: https://doi.org/10.1016/j.patter.2022.100488
	* 'doping': Named entity recognition for solid-state doping [DOPANT (dopant species), BASEMAT (host material), DOPMODQ (dopant quantity or carrier density)] using BERT pre-trained on materials science text.
		 - More info: https://doi.org/10.1016/j.patter.2022.100488
	* 'solid_state': Named entity recognition for solid state materials data using BERT pre-trained on materials science text.
		 - More info: https://doi.org/10.1016/j.patter.2022.100488

```

Here, we see there are two models packaged together in the `matscholar_2020v1` package, one in `relevance_2020v1`, and four are packaged together in `matbert_ner_2021v1`. The model package names and the model names are what we will use to load them in the following code.

## Load a pretrained model (NER)

Model downloads and file management are done automatically. You only need to call a single `load` function from the model package module
in order to load and deploy the model locally.


```python

# Import the load function from the model package
from lbnlp.models.load.matscholar_2020v1 import load

ner_model = load("ner")
```


*If you get a `ModelReqirementError`, we recommend installing the package (with specific version) listed in the error. This is to avoid silent errors and annoying errors, as our models were generated with specific dependencies.*


Our NER model is now loaded. Let's see what it can do!

```python
doc = "CoCrPt/CoCr/carbon films were sputter-deposited on CoTaZr soft-magnetic" \
    "underlayers and the effects of a carbon intermediate layer on magnetic and " \
    "recording properties were investigated"

tags = ner_model.tag_doc(doc)
print(tags)
```


The tokens in the doc are annotated:
```
[[('CoCrPt', 'B-MAT'), ('/', 'O'), ('CoCr', 'B-MAT'), ('/', 'O'), ('carbon', 'B-MAT'),
 ('films', 'B-DSC'), ('were', 'O'), ('sputter', 'B-SMT'), ('-', 'I-SMT'),
('deposited', 'I-SMT'), ('on', 'O'), ('CoTaZr', 'B-MAT'), ('soft', 'B-PRO'),
('-', 'I-PRO'), ('magnetic', 'I-PRO'), ('underlayers', 'I-PRO'), ('and', 'O'),
 ('the', 'O'), ('effects', 'O'), ('of', 'O'), ('a', 'O'), ('carbon', 'B-MAT'),
 ('intermediate', 'O'), ('layer', 'B-DSC'), ('on', 'O'), ('magnetic', 'B-PRO'),
('and', 'O'), ('recording', 'B-PRO'), ('properties', 'I-PRO'), ('were', 'O'),
('investigated', 'O')]]
```

## Working with models


Since many of the models were designed, trained, and deployed by different authors with different goals, consult the documentation for the specific model you are interested in under the "Pretrained" section of this documentation.

Each model has its own unique methods to call to obtain the desired results, so follow the guides on the individual pages for the best results.


**Each model package will have it's own specific requirements!**. You can find these in the root `lbnlp` directory as `.txt` files interpretable by `pip`, and alternatively in the model package metadata - we will do our best to warn you if a required package is not found before loading a model.

