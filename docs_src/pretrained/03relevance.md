# Models - Relevance Classification


## Load a model for Relevance Classification

Load the relevance classification model from `matscholar_2020v1` by calling:

```python

from lbnlp.models.load.relevance_2020v1 import load

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

Note that the relevance classifier works better with longer pieces of text.



```python
from lbnlp.models.load.relevance_2020v1 import load

clf_model = load("relevance", ignore_requirements=True)

matsci1 = "It is very difficult and yet extremely important to fill the wide technological gap in developing transparent conducting oxides (TCOs) that exhibit excellent p-type conducting characteristics. Here, on the basis of extensive first-principles calculations, we discover for the first time potentially promising p-type transparent conductivity in Zn-doped TiO2 under oxygen rich conditions. Efforts have been made to elaborate the effects of possible defects and their interaction with Zn doping on the p-type transparent conductivity. This work offers a fundamental road map for cost-effective development of p-type TCOs based on TiO2, which is a cheap and stable material system of large natural resources."
matsci2 = "Nb doped TiO2 (Nb:TiO2) is a promising indium-free transparent conducting oxide. We have examined the growth of Nb:TiO2 thin films by pulsed laser deposition (PLD) on SrTiO3, LaAlO3, and fused silica. For <004> oriented anatase Nb:TiO2 films grown on SrTiO3 by PLD at 550 °C, the conductivity can be as high as 2500 S/cm. A nearly thickness independent conductivity for Nb:TiO2 demonstrates that the conductivity is a bulk property and not a substrate interface effect. In addition, Nb:TiO2 films deposited at room temperature were annealed at temperatures up to 750 °C in either vacuum or 1.3×10−3 Pa O2. For these films, conductivities as high as 3300 S/cm on SrTiO3 and 85 S/cm on LaAlO3 substrates were obtained for the highest temperature vacuum anneals, albeit with some loss in transparency."
bio = "Gene therapy has long held promise to correct a variety of human diseases and defects. Discovery of the Clustered Regularly-Interspaced Short Palindromic Repeats (CRISPR), the mechanism of the CRISPR-based prokaryotic adaptive immune system (CRISPR-associated system, Cas), and its repurposing into a potent gene editing tool has revolutionized the field of molecular biology and generated excitement for new and improved gene therapies. Additionally, the simplicity and flexibility of the CRISPR/Cas9 site-specific nuclease system has led to its widespread use in many biological research areas including development of model cell lines, discovering mechanisms of disease, identifying disease targets, development of transgene animals and plants, and transcriptional modulation. In this review, we present the brief history and basic mechanisms of the CRISPR/Cas9 system and its predecessors (ZFNs and TALENs), lessons learned from past human gene therapy efforts, and recent modifications of CRISPR/Cas9 to provide functions beyond gene editing. We introduce several factors that influence CRISPR/Cas9 efficacy which must be addressed before effective in vivo human gene therapy can be realized. The focus then turns to the most difficult barrier to potential in vivo use of CRISPR/Cas9, delivery. We detail the various cargos and delivery vehicles reported for CRISPR/Cas9, including physical delivery methods (e.g. microinjection; electroporation), viral delivery methods (e.g. adeno-associated virus (AAV); full-sized adenovirus and lentivirus), and non-viral delivery methods (e.g. liposomes; polyplexes; gold particles), and discuss their relative merits. We also examine several technologies that, while not currently reported for CRISPR/Cas9 delivery, appear to have promise in this field. The therapeutic potential of CRISPR/Cas9 is vast and will only increase as the technology and its delivery improves."
random = "As an action research project, using mixed methodology, this study investigated how the use of math journals affected second grade students’ communication of mathematical thinking. For this study, math journal instruction was provided. The data gathering included pre- and post- math assessment, students’ math journals, interviews with the students, and teacher’s reflective journal. Findings of the study indicated that the use of math journals positively influenced the students’ communication of mathematical thinking and the use of math vocabulary. Additionally, math journals served as a communication tool between the students and teacher and an assessment tool for the teacher. The implications of this study regarding students’ writing ability and time constraints issues were also discussed."

relevance = clf_model.classify_many([matsci1, bio, random, matsci2])
# should print [1 0 0 1]
print(relevance)
```

Output
```
[1 0 0 1]
```
