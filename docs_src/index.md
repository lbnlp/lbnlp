# About LBNLP

![icon](static/lbl_logo_cropped.jpg)

LBNLP is a joint natural language processing (NLP) effort between the [Ceder group](https://ceder.berkeley.edu/), [HackingMaterials Group](https://hackingmaterials.lbl.gov), and [Persson Group](https://perssongroup.lbl.gov/) at Lawrence Berkeley National Laboratory natural aimed at **materials science text.**




### LBNLP is currently being migrated and condensed from several source repos, and it is currently in an experimental state.


That being said, some of the things you can currently do with LBNLP are:


### Preprocess and tokenize text with methods specific to materials science

The `MatScholarProcess` class implements methods specific to materials science and chemistry, including phrasing of specific materials-science lingo.



### Load pre-trained models in 2 lines of code


LBNLP also gives access to open-source pre-trained NLP models such as:

- BiLSTM-based named entity recognition for solid state materials entities (such as characterization methods)
- BERT-based named entity recognition (MatBERT) for general solid state materials
- BERT-based named entity recognition (MatBERT) for specific downstream tasks (identifying Au nanoparticle morphologies, identifying dopants and host materials)
- Logistic regression/TFIDF-based relevance classification for identifying whether text is relevant specifically to solid-state materials science


With the following models to come:
- MatBERT for doping analysis using an improved annotation scheme for improved accuracy
- Doc2Vec for suggesting journals based on input text alone
- Access to Mat2Vec embeddings from the [original Mat2Vec paper](https://doi.org/10.1038/s41586-019-1335-8)



**Each of these models is condensed to an extremely simple interface (1 method to run inference) can be loaded in 2 lines of code.** 


See the Pretrained documentation on the side bar for more information on how to run, load, and interpret these models.