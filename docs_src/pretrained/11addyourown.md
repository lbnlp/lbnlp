# Adding your own trained models to LBNLP

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