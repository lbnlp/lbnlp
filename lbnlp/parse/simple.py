import re
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition


class SimpleParser:
    '''
    A parser class to identify chemical mentions, and related them
    to their canonical (normalized, alphabetized) form.

    Can handle formulae containing brackets and fractional occupations.
    Example inputs:
    LiFePO4
    Li2(FePO4)2
    Sr(Zr0.5Ti0.5)O3

    Output is a defaultdict() object that is normalized and alphabetized.
    Examples:

    >>> parser = SimpleParser()
    >>> parser.parse(Li2(FePO4)2)
    'FeLiO4P'
    >>> parser.parse(Sr(Zr0.5Ti0.5)O3)
    'O6Sr2TiZr'
    '''

    def __init__(self):
        self.name = "ImprovedMaterialParser"

    def is_element(self, element):
        '''
        Checks if element is a chemical symbol.
        '''
        try:
            Element(element)
            return True
        except:
            return False

    def alphabetize(self, formula):
        '''
        Take a chemical formula such as SrZrO3 and returns alphabetized version O3SrZr.
        '''
        return ''.join(sorted(re.findall(r'[A-Z][a-z]?\d*', formula)))

    def matgen_parser(self, formula):
        '''
        Converts formula string to canonical (normalized, alphabetized) form.
        Returns defaultdict() object containing formula if successful. Returns false
        if an exception is raised.
        '''
        try:
            integer_formula, factor = Composition(formula).get_integer_formula_and_factor()
            composition = Composition(integer_formula)
            if any([not self.is_element(key) for key in composition.keys()]):
                return False
            else:
                reduced = composition.get_reduced_formula_and_factor()[0]
                ordered = self.alphabetize(reduced)
                return ordered
        except:
            return False


    def parse(self, cem):
        '''
        Parses and returns formula.
        '''
        parsers = [self.matgen_parser]
        parsed = False
        for parser in parsers:
            if parser(cem):
                parsed = parser(cem)
                break
        if parsed:
            return parser(cem)
        else:
            return False