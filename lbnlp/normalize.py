import json
import os
import re
from lbnlp.parse.material import MaterialParser
from lbnlp.parse.simple import SimpleParser
from chemdataextractor.doc import Paragraph


class Normalizer:
    """
    A class to perform entity normalization.
    """

    def __init__(self, data_path, material_parser_data_path):
        """
        Constructor method for Normalizer.
        """

        self.normal_dict = NormalDict(data_path)
        self.mat_normalizer = MatNormalizer(data_path, material_parser_data_path)

    # TODO: Make this normalize a single document
    def normalize(self, raw_docs, tagged_docs):
        """
        Normalize the entities in a list of documents.

        :param raw_docs: a list of strings
        :param tagged_docs: a list of documents; each document is a list of sentences;
        each sentence is a list of tuples of the form (word, iob-tag)
        :return: a list of normalized documents
        """

        normalized_docs = []
        for tagged, raw in zip(tagged_docs, raw_docs):
            normalized_doc = []
            concatenated = self._concatenate_ents(tagged)
            all_mats = [entity for sent in tagged for entity,
                        tag in sent if tag == 'MAT']
            for sent in concatenated:
                normalized_sent = []
                for ent, tag in sent:
                    normalized_ent = self._normalize_ents(
                        ent, tag, all_mats, raw)
                    normalized_sent.append((normalized_ent, tag))
                normalized_doc.append(normalized_sent)
            normalized_docs.append(normalized_doc)
        return normalized_docs

    def _normalize_ents(self, ent, tag, all_mats, raw_text):
        """
        Normalizes each type of entity

        :param ent:
        :param tag:
        :param all_mats:
        :param raw_text:
        :return:
        """
        if tag in self.normal_dict.DICT_KEYS:
            return self.normal_dict[tag][ent]["most_common"] if ent in self.normal_dict[tag] else ent
        elif tag == "MAT":
            return self.mat_normalizer.normalize_mat(ent, all_mats, raw_text)
        else:
            return ent

    @staticmethod
    def _concatenate_ents(doc_as_iob):
        """
        Takes multiword entities in iob format (token, iob) and return

        :param doc_as_iob: list; a document in iob format; the doscument should be a list of
        sentences; each sentence is a list of tuples of the form (word, iob-tag)
        :return: list; document with concatenated entity tags
        """
        new_doc = []
        named_entity = []
        tag_type = None
        for sent in doc_as_iob:
            new_sent = []
            for word, tag in sent:
                if not named_entity and tag[0] == "B":
                    named_entity = [word]
                    tag_type = tag[-3:]
                elif named_entity and tag[0] == "I":
                    named_entity.append(word)
                elif named_entity and tag[0] != "I":
                    new_ent = " ".join(named_entity)
                    new_sent.append((new_ent, tag_type))
                    if tag == "O":
                        new_sent.append((word, tag))
                        named_entity = []
                    else:
                        named_entity = [word]
                        tag_type = tag[-3:]
                else:
                    new_sent.append((word, tag))
            new_doc.append(new_sent)
        return new_doc


class NormalDict(dict):
    """
    A dictionary for normalizing entities
    """

    DICT_KEYS = [
        "PRO",
        "APL",
        "DSC",
        "SPL",
        "SMT",
        "CMT"
    ]

    def __init__(self, data_path):
        """
        A dictionary for mapping entities onto their normalized form
        """

        super().__init__()
        for key in self.DICT_KEYS:
            dict_path = os.path.join(data_path, f"{key.lower()}.json")
            with open(dict_path, 'r') as f:
                dict_ = json.load(f)
            self[key] = {k.upper(): v for k, v in dict_.items()}

    def __getitem__(self, item):
        if item in self.DICT_KEYS:
            return dict.__getitem__(self, item)
        else:
            raise KeyError("Invalid key {} for NormalDict".format(item))


class MatNormalizer:
    """
    A class for material normalization. Where possible, each material mention is
    converted to a chemical formula that is alphabetized and divided by the lowest
    common factor.
    """

    GREEK_LETTERS = ''.join([chr(i) for i in range(945, 970)])
    SPECIAL_MATERIALS = [
        "diamond",
        "graphene",
        "graphite",
        "fullerene",
        "fullerenes",
        "amorphous carbon",
        "glassy carbon"
        "diamond-like carbon",
        "carbon nanotube",
        "carbon nanotubes",
        "lonsdaleite",
        "silicene",
        "steel",
        "cement",
        "concrete",
        "stainless steel",
        "black phosphorous",
        "red phosphorous",
        "white phosphorous",
        "phosphorene",
        "graphene oxide",
        "oxides",
        "nitrides",
        "phosphides",
        "arsenides",
        "oxide",
        "nitride",
        "phosphide",
        "arsenide"
    ]

    def __init__(self, data_path, material_parser_data_path):
        """
        Constructor method for MatNormalizer.
        """
        self.mp = MaterialParser(data_path=material_parser_data_path)
        self.mp_lookup = MaterialParser(data_path=material_parser_data_path, pubchem_lookup=True)
        self.matgen_parser = SimpleParser().matgen_parser

        with open(os.path.join(data_path, "mat2formula.json")) as f:
            mat_lookup = json.load(f)
        self.mat_lookup = mat_lookup

    def normalize_mat(self, mat, all_mats, raw_text):
        """
        Normalizes a material mention to a canonical chemical formala

        :param mat: string; the material mention to be normalized
        :param all_mat: list; all of the materials extracted from raw_text
        :param raw_text: string; the raw text form which the material was extracted
        :return: string; normalized formula (alphabetized and divided by the highest common factor)
        """

        # Make sure multitokens are separated by spaces not underscores
        mat = mat.replace("_", " ")
        all_mats = [mat_.replace("_", " ") for mat_ in all_mats]

        # If material is a special material, don't normalize
        if mat in self.SPECIAL_MATERIALS:
            return mat

        # Check if formula can be directly normalized
        matgen_normalized = self.matgen_parser(mat)
        if matgen_normalized:
            return matgen_normalized

        # Lookup table
        if mat in self.mat_lookup:
            return self.mat_lookup[mat]

        # Check for an acronym
        acronym = self.get_acronyms(mat, all_mats, raw_text)
        if acronym:
            return acronym

        # Check for wildcards
        wildcards = self.get_wildcards(mat, raw_text)
        if wildcards:
            return wildcards

        # Check for fractions
        fractions = self.get_fractions(mat, raw_text)
        if fractions:
            return fractions

        return mat

    def get_acronyms(self, mat, all_mats, raw_text):
        """
        Converts a material acronym to a full stoichiometry.
        Example: "STO" ---> "SrTiO3"

        :param mat: string; a material acronym
        :param all_mats: list; all of the materials extracted from the document
        :param raw_text: string; raw text of the document from which the material was extracted
        :return: string; a normalized stoichiometry if found, else None
        """
        acronyms = self.mp.build_abbreviations_dict(all_mats, raw_text)
        try:
            parsed_mat = self.mp.get_chemical_structure(acronyms[mat])[
                "formula"]
            matgen_normalized = self.matgen_parser(parsed_mat)
        except KeyError:
            return
        return matgen_normalized

    def get_wildcards(self, mat, raw_text):
        """
        Resolves material wild cards.
        Example: "AO2 (A = Ti, Zr)" ---> ["O2Ti", "O2Zr"]

        :param mat: string; a material containing a wildcard
        :param raw_text: string; raw text of the document from which the material was extracted
        :return: list; a list containing the resolved stoichiometries if found, else None
        """

        regexpr = r"((?:[A-Z][a-z]?\d*[A-Za-z0-9" + \
            self.GREEK_LETTERS + "\(\)\+\-]*){2,})"
        formula = re.findall(regexpr, mat)
        if not formula:
            return
        try:
            struct = self.mp.get_chemical_structure(formula[0])
        except:
            return
        return self._wildcards(struct, raw_text)

    def get_fractions(self, mat, raw_text):
        """
        Resolves materials fractions represented by placeholders.
        Example: "BaxSr(1-x)TiO3 (x = 0.25, 0.5)" ---> ["BaO12Sr3Ti4", "BaO6SrTi2"]

        :param mat: string
        :param raw_text:
        :return: list; a list containing the resolved stoichiometries if found, else None
        """

        regexpr = r"((?:[A-Z][a-z]?\d*[A-Za-z0-9" + \
            self.GREEK_LETTERS + "\(\)\+\-]*){2,})"
        formula = re.findall(regexpr, mat)
        if not formula:
            return
        try:
            struct = self.mp.get_chemical_structure(formula[0])
            parsed_fractions = self._fractions(struct, raw_text)
        except:
            return
        parsed_fractions = [pf for pf in parsed_fractions if pf]
        el_in_formula = len(re.findall(r"[A-Z][a-z]?", formula[0]))
        if any(len(re.findall(r"[A-Z][a-z]?", frac)) < el_in_formula - 1 for frac in parsed_fractions):
            return
        return parsed_fractions

    def _wildcards(self, chemical_structure, raw_text):
        el_v = chemical_structure["elements_vars"].keys()
        new_compositions = []
        for v in el_v:
            for to_sub in self.mp.get_elements_values(v, raw_text):
                new_compositions.append(
                    chemical_structure["formula"].replace(v, to_sub))
        normalized_compositions = []
        for composition in new_compositions:
            normalized_composition = self.matgen_parser(composition)
            if normalized_composition:
                normalized_compositions.append(normalized_composition)
        return normalized_compositions

    def _fractions(self, chemical_structure, raw_text):
        f_v = chemical_structure["fraction_vars"].keys()
        composition = chemical_structure["composition"]
        new_compositions = []
        for v in f_v:
            values = self._find_variables(v, raw_text, self.mp)
            for value in values:
                comp = {key: str(eval(_value.replace(v, str(value))))
                        for key, _value in composition.items()}
                comp = self.matgen_parser(
                    ''.join([key + _value if _value != str(1) else key for key, _value in comp.items()]))
                new_compositions.append(comp)
        return new_compositions

    def _find_variables(self, var, raw_text, mp):
        sents = Paragraph(raw_text).sentences
        i = 0
        values = []
        while len(values) == 0 and i < len(sents):
            sent = sents[i]
            try:
                values, mode = mp.get_stoichiometric_values(var, sent.text)
            except ValueError:
                pass
            i += 1
        return values
