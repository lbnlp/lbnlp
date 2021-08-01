# Models - Relevance Classification


## Load a model for Relevance Classification

Load the relevance classification model from `matscholar_2020v1` by calling:

```python

from lbnlp.models.load.matscholar_2020v1 import load

clf_model = load("relevance")
```

Let's see how it does in determining relevance for inorganic materials science:

```python

not_relevant = "The polymer was used for an OLED. This can also be used for a biosensor."
relevant = "The bandgap of ZnO is 33 eV"

relevance = clf_model.classify_many([not_relevant, relevant])
print(relevance)
```

It correctly classifies our sentences:

```
[0, 1]
```
