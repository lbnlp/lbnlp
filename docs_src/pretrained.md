
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
	* 'relevance': Relevance classification with logistic regression for materials science abstracts.
		 - More info: None
```

Here, we see there are two models packaged together in the `matscholar_2020v1` package. The names of the model package and the models are what we will use to load them in the following code.


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
doc = "CoCrPt/CoCr/carbon films were sputter-deposited on CoTaZr soft-magnetic"
    "underlayers and the effects of a carbon intermediate layer on magnetic and "
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


## Load a different model (Relevance Classification)

Load another model from the same model package by calling `load` with it's name:

```python

from lbnlp.models.load.matscholar_2020v1 import load

clf_model = load("relevance")
```

Let's see how it does in determining relevance for inorganic materials science:

```python

not_relevant = "The polymer was used for an OLED. This can also be sued for a biosensor."
relevant = "The bandgap of ZnO is 33 eV"

relevant = clf_model.classify_many([not_relevant, relevant])
print(relevant)
```

It correctly classifies our sentences:

```
[0, 1]
```
---

## Adding your own trained models to LBNLP

#### 1. Store your data

Using whatever directory structure you want, store your model(s)' data in some loadable format. 

#### 2. Upload your data as a file

Zip that directory and upload it to a publicly accessible URL.

#### 3. Make an modelpkg entry 

Make an entry for a new model package in `lbnlp/models/modelpkg_metadata.json`, including all needed requirements specific to your models, citations, and descriptions.

```json
  "my_model_package_2021v2": {
    "url": "https://url.to.download/your/file",
    "hash": "The SHA256 hash of your zip file",
    "models": {
      "my_first_model": {
        "requirements": ["pip-accessible-requirement==1.0.0"],
        "description": "Model 1, a good description",
        "citation": null
      },
      "my_second_model": {
        "requirements": ["sklearn==0.19.1"],
        "description": "Model 2, an even better model!",
        "citation": null
```


#### 4. Add a data loader

Create a new module in `lbnlp.models.load` with the same name you gave your package in the metadata.

This module is where you will put the functions to load your models.

Follow the example below to make full use of the helper functions and classes:

```python


from lbnlp.models.fetch import ModelPkgLoader
from lbnlp.models.util import model_loader_setup

# Provides an easy way to access file paths and metadata for your package
pkg = ModelPkgLoader("my_model_package_2021v2")

# Load is the required name for the loading function.
# Keep this function signature the same as shown here
# model_loader_setup provides checking and validation in the background

@model_loader_setup(pkg)
def load(model_name, ignore_requirements=False):

    # This is the root of the directory of the zip file you uploaded.
    base_path = pkg.structured_path

    # Your code for loading your model goes here
```

The model loader code should only access classes and methods in lbnlp or in the requirements. Make it easy for people to load and use your model!

That's it! Congrats on adding your model to LBNLP.