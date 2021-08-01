
# Models - Named Entity Recognition

## BiLSTM-NER for Solid State Materials Data
### `matscholar_2020v1` - `ner`

The BiLSTM-NER model tags inorganic solid-state entities in materials science. Read more details in [the original publication.](https://doi.org/10.1021/acs.jcim.9b00470)


We can import this model from the `matscholar_2020v1` model package.

*Note: this model requires a specific, older version (1.15.0) of tensorflow in order to run.*

```python
# Import the load function from the model package
from lbnlp.models.load.matscholar_2020v1 import load

ner_model = load("ner")
```


Our NER model is now loaded. Let's annotate a document.

```python
doc = "CoCrPt/CoCr/carbon films were sputter-deposited on CoTaZr soft-magnetic"
    "underlayers and the effects of a carbon intermediate layer on magnetic and "
    "recording properties were investigated"

tags = ner_model.tag_doc(doc)
print(tags)
```


The tokens in the doc are annotated as tuples according to the IOB (inside entity `I`, outside entity `O`, beginning of entity `B`) scheme to handle multi-token entities and the following entity tags

- `MAT`: material
- `DSC`: description of sample
- `SPL`: symmetry or phase label
- `SMT`: synthesis method
- `CMT`: characterization method
- `PRO`: property - may also include `PVL` (property value) or `PUT` (property unit)
- `APL`: application


An example tag would be `B-MAT` meaning beginning of a material entity followed by `I-MAT` meaning a continuation of that material entity.

The output of our code is a list of sentences - each sentence is a list of tuples coupling the text with the label:
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



## BiLSTM-NER - Simpler model
### `matscholar_2020v1` - `ner_simple`

There is an alternative model for NER which simplifies the annotation scheme to not include `PVL` or `PUT` and which
merges `B`/`I` multitoken entities into contiguous strings. To use it, use the `ner_simple` model from `matscholar_2020v1`:


```python
from lbnlp.models.load.matscholar_2020v1 import load

ner_simple = load("ner_simple")
```

This model has the same `tag_doc` interface as the original `ner` model.

```python
doc = "CoCrPt/CoCr/carbon films were sputter-deposited on CoTaZr soft-magnetic"
    "underlayers and the effects of a carbon intermediate layer on magnetic and "
    "recording properties were investigated"

tags = ner_simple.tag_doc(doc)
print(tags)
```


The output joins the IOB scheme into only the 7 entity types. The rest of the format is identical to the original `ner` model:


```stdout
[[('CoCrPt', 'MAT'), ('/', 'O'), ('CoCr', 'MAT'), ('/', 'O'), ('carbon', 'MAT'), 
('films', 'DSC'), ('were', 'O'), ('sputter - deposited', 'SMT'), ('on', 'O'), 
('CoTaZr', 'MAT'), ('soft - magneticunderlayers', 'PRO'), ('and', 'O'), ('the', 'O'), 
('effects', 'O'), ('of', 'O'), ('a', 'O'), ('carbon', 'MAT'), ('intermediate', 'O'), 
('layer', 'DSC'), ('on', 'O'), ('magnetic', 'PRO'), ('and', 'O'), 
('recording properties', 'PRO'), ('were', 'O'), ('investigated', 'O')]]
```




## MatBERT-NER for Solid State, Gold Nanoparticle, and Dopant data
### `matbert_ner_2021v1` - `solid_state`, `aunp2`, `aunp11`, and `doping`

Let's load a more involved MatBERT model and use it for an example:

```python

from lbnlp.models.load.matbert_ner_2021v1 import load

bert_ner = load("solid_state")

```


Let's see if we can tag an excerpt from a materials science abstract.

The MatBERT-NER model works with lists of texts as input, returning lists of summary documents as output.


```python



doc = "Synthesis of carbon nanotubes by chemical vapor deposition over patterned " \
      "catalyst arrays leads to nanotubes grown from specific sites on surfaces."  \
      "The growth directions of the nanotubes can be controlled by van der Waals " \
      "self-assembly forces and applied electric fields. The patterned growth " \
      "approach is feasible with discrete catalytic nanoparticles and scalable " \
      "on large wafers for massive arrays of novel nanowires." 

# the MatBERT model is intended to be used with batches (multiple documents at once)
# so we just put the doc into a list before tagging
tags = bert_ner.tag_docs([doc])


print(tags)

```


We obtain a summary document for this abstract based on the extracted entities:

```stdout
[{'entities': {'APL': ['nanotechnology',
                       'catalyst',
                       'photochemistry',
                       'molecular sensors',
                       'catalytic',
                       'devices',
                       'chemical functionalization',
                       'nanoscience'],
               'CMT': [],
               'DSC': ['nanotubes',
                       'wafers',
                       'surfaces',
                       'nanowires',
                       'nanoparticles'],
               'MAT': ['carbon'],
               'PRO': ['electromechanical properties',
                       'electrical',
                       'mechanical',
                       'surface chemistry'],
               'SMT': ['chemical vapor deposition'],
               'SPL': []},
  'tokens': [[{'annotation': 'O', 'text': 'synthesis'},
              {'annotation': 'O', 'text': 'of'},
              {'annotation': 'MAT', 'text': 'carbon'},
              {'annotation': 'DSC', 'text': 'nanotubes'},
              {'annotation': 'O', 'text': 'by'},
              {'annotation': 'SMT', 'text': 'chemical'},
              {'annotation': 'SMT', 'text': 'vapor'},
              {'annotation': 'SMT', 'text': 'deposition'},
              {'annotation': 'O', 'text': 'over'},
              {'annotation': 'O', 'text': 'patterned'},
              {'annotation': 'APL', 'text': 'catalyst'},
              {'annotation': 'O', 'text': 'arrays'},
              {'annotation': 'O', 'text': 'leads'},
              {'annotation': 'O', 'text': 'to'},
              {'annotation': 'DSC', 'text': 'nanotubes'},
              {'annotation': 'O', 'text': 'grown'},
              {'annotation': 'O', 'text': 'from'},
              {'annotation': 'O', 'text': 'specific'},
              {'annotation': 'O', 'text': 'sites'},
              {'annotation': 'O', 'text': 'on'},
              {'annotation': 'DSC', 'text': 'surfaces'},
              {'annotation': 'O', 'text': '.'}],
             [{'annotation': 'O', 'text': 'the'},
              {'annotation': 'O', 'text': 'growth'},
              {'annotation': 'O', 'text': 'directions'},
              {'annotation': 'O', 'text': 'of'},
              {'annotation': 'O', 'text': 'the'},
              {'annotation': 'DSC', 'text': 'nanotubes'},
              {'annotation': 'O', 'text': 'can'},
              {'annotation': 'O', 'text': 'be'},
              {'annotation': 'O', 'text': 'controlled'},
              {'annotation': 'O', 'text': 'by'},
              {'annotation': 'O', 'text': 'van'},
              {'annotation': 'O', 'text': 'der'},
              {'annotation': 'O', 'text': 'waals'},
              {'annotation': 'O', 'text': 'self'},
              {'annotation': 'O', 'text': '-'},
              {'annotation': 'O', 'text': 'assembly'},
              {'annotation': 'O', 'text': 'forces'},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'O', 'text': 'applied'},
              {'annotation': 'O', 'text': 'electric'},
              {'annotation': 'O', 'text': 'fields'},
              {'annotation': 'O', 'text': '.'}],
             [{'annotation': 'O', 'text': 'the'},
              {'annotation': 'O', 'text': 'patterned'},
              {'annotation': 'O', 'text': 'growth'},
              {'annotation': 'O', 'text': 'approach'},
              {'annotation': 'O', 'text': 'is'},
              {'annotation': 'O', 'text': 'feasible'},
              {'annotation': 'O', 'text': 'with'},
              {'annotation': 'O', 'text': 'discrete'},
              {'annotation': 'APL', 'text': 'catalytic'},
              {'annotation': 'DSC', 'text': 'nanoparticles'},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'O', 'text': 'scalable'},
              {'annotation': 'O', 'text': 'on'},
              {'annotation': 'O', 'text': 'large'},
              {'annotation': 'DSC', 'text': 'wafers'},
              {'annotation': 'O', 'text': 'for'},
              {'annotation': 'O', 'text': 'massive'},
              {'annotation': 'O', 'text': 'arrays'},
              {'annotation': 'O', 'text': 'of'},
              {'annotation': 'O', 'text': 'novel'},
              {'annotation': 'DSC', 'text': 'nanowires'},
              {'annotation': 'O', 'text': '.'}],
             [{'annotation': 'O', 'text': 'controlled'},
              {'annotation': 'O', 'text': 'synthesis'},
              {'annotation': 'O', 'text': 'of'},
              {'annotation': 'DSC', 'text': 'nanotubes'},
              {'annotation': 'O', 'text': 'opens'},
              {'annotation': 'O', 'text': 'up'},
              {'annotation': 'O', 'text': 'exciting'},
              {'annotation': 'O', 'text': 'opportunities'},
              {'annotation': 'O', 'text': 'in'},
              {'annotation': 'APL', 'text': 'nanoscience'},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'APL', 'text': 'nanotechnology'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'O', 'text': 'including'},
              {'annotation': 'PRO', 'text': 'electrical'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'PRO', 'text': 'mechanical'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'PRO', 'text': 'electromechanical'},
              {'annotation': 'PRO', 'text': 'properties'},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'APL', 'text': 'devices'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'APL', 'text': 'chemical'},
              {'annotation': 'APL', 'text': 'functionalization'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'PRO', 'text': 'surface'},
              {'annotation': 'PRO', 'text': 'chemistry'},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'APL', 'text': 'photochemistry'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'APL', 'text': 'molecular'},
              {'annotation': 'APL', 'text': 'sensors'},
              {'annotation': 'O', 'text': ','},
              {'annotation': 'O', 'text': 'and'},
              {'annotation': 'O', 'text': 'interfacing'},
              {'annotation': 'O', 'text': 'with'},
              {'annotation': 'O', 'text': 'soft'},
              {'annotation': 'O', 'text': 'biological'},
              {'annotation': 'O', 'text': 'systems'},
              {'annotation': 'O', 'text': '.'}]]}]

```

The `entities` key represents a summary of each entity found in the document.
The `tokens` key contains a dictionary of each token and its associated predicted label, separated into sentences by lists.


The other models such as `doping` (3-tag scheme), `aunp2` (2-tag scheme gold nanoparticle), and `aunp11` (11-tag scheme for gold nanoparticle)
will have more explanation in an upcoming publication.