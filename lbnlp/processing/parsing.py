import re
import collections
import sympy
from sympy.abc import _clash
from chemdataextractor.doc import Document
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition, CompositionError
import pubchempy as pcp


class MaterialParser:
    def __init__(self):
        self.__list_of_elements_1 = ['H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'K', 'V', 'Y', 'I', 'W', 'U']
        self.__list_of_elements_2 = ['He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ar', 'Ca', 'Sc', 'Ti', 'Cr',
                                     'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                                     'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe',
                                     'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
                                     'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
                                     'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                                     'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn',
                                     'Fl', 'Lv']
        self.__list_of_trash_words = ['bulk', 'coated', 'rare', 'earth', 'ceramics', 'undoped']
        self.__greek_letters = ['α', 'δ', 'χ']

    ###################################################################################################################
    ### Methods to build chemical structure
    ###################################################################################################################

    def __simplify(self, value):

        """
        simplifying stoichiometric expression
        :param value: string
        :return: string
        """

        for l in self.__greek_letters:
            _clash[l] = sympy.Symbol(l)

        new_value = value
        for i, m in enumerate(re.finditer('(?<=[0-9])([a-z'+''.join(self.__greek_letters)+'])', new_value)):
            new_value = new_value[0:m.start(1) + i] + '*' + new_value[m.start(1) + i:]
        new_value = sympy.simplify(sympy.sympify(new_value, _clash))
        if new_value.is_Float:
            new_value = round(float(new_value), 3)

        return str(new_value)

    def __get_sym_dict(self, f, factor):
        sym_dict = collections.defaultdict(str)
        r = "([A-Z]{1}[a-z]{0,1})\s*([-*\.\da-z"+''.join(self.__greek_letters)+"\+/]*)"

        for m in re.finditer(r, f):
            """
            checking for correct elements names
            """
            el_bin = "{0}{1}".format(str(int(m.group(1)[0] in self.__list_of_elements_1 + ['M'])), str(
                int(m.group(1) in self.__list_of_elements_1 + self.__list_of_elements_2 + ['Ln', 'M'])))
            if el_bin in ['01', '11']:
                el = m.group(1)
                amt = m.group(2)
            if el_bin in ['10', '00']:
                el = m.group(1)[0]
                amt = m.group(1)[1:] + m.group(2)

            if len(sym_dict[el]) == 0:
                sym_dict[el] = "0"
            if amt.strip() == "":
                amt = "1"
            sym_dict[el] = '(' + sym_dict[el] + ')' + '+' + '(' + amt + ')' + '*' + '(' + str(factor) + ')'
            f = f.replace(m.group(), "", 1)
        if f.strip():
            return collections.defaultdict(str)
            # print("{} is an invalid formula!".format(f))

        """
        refinement of non-variable values
        """
        for el, amt in sym_dict.items():
            sym_dict[el] = self.__simplify(amt)

        return sym_dict

    def parse_formula(self, formula):
        """
        Args:
            formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3
        Returns:
            Composition with that formula.
        """
        formula_dict = collections.defaultdict(str)
        r = "\(([^\(\)]+)\)\s*([-*\.\da-z"+''.join(self.__greek_letters)+"\+/]*)"

        while len(re.findall(r, formula)) > 0:
            for m in re.finditer(r, formula):
                factor = "1"
                if m.group(2) != "":
                    factor = m.group(2)
                unit_sym_dict = self.__get_sym_dict(m.group(1), factor)
                for el in formula_dict:
                    formula_dict[el] = '(' + formula_dict[el] + ')' + '*' + factor
                for el, amt in unit_sym_dict.items():
                    if len(formula_dict[el]) == 0:
                        formula_dict[el] = amt
                    else:
                        formula_dict[el] = '(' + formula_dict[el] + ')' + '+' + '(' + amt + ')'
                formula = formula.replace('(' + m.group(1) + ')' + m.group(2), '', 1)

        # if there is coefficient before formula, change factor
        unit_sym_dict = self.__get_sym_dict(formula, "1")
        for el, amt in unit_sym_dict.items():
            if len(formula_dict[el]) == 0:
                formula_dict[el] = amt
            else:
                formula_dict[el] = '(' + formula_dict[el] + ')' + '+' + '(' + amt + ')'

        """
        refinement of non-variable values
        """
        incorrect = []
        for el, amt in formula_dict.items():
            formula_dict[el] = self.__simplify(amt)
            if any(len(c) > 1 for c in re.findall('[A-Za-z]+', formula_dict[el])):
                incorrect.append(el)

        for el in incorrect:
            del formula_dict[el]

        return formula_dict

    def get_structure_by_formula(self, formula):

        init_formula = formula
        formula = formula.replace(' ', '')
        formula = formula.replace('−', '-')
        formula = formula.replace('[', '(')
        formula = formula.replace(']', ')')

        elements_variables = collections.defaultdict(str)
        stoichiometry_variables = collections.defaultdict(str)

        # check for any weird syntax
        r = "\(([^\(\)]+)\)\s*([-*\.\da-z\+/]*)"
        for m in re.finditer(r, formula):
            if not m.group(1).isupper() and m.group(2) == '':
                formula = formula.replace('(' + m.group(1) + ')', m.group(1), 1)
            if ',' in m.group(1):
                elements_variables['M'] = re.split(',', m.group(1))
                formula = formula.replace('(' + m.group(1) + ')' + m.group(2), 'M' + m.group(2), 1)

        composition = self.parse_formula(formula)

        # looking for variables in elements and stoichiometry
        for el, amt in composition.items():
            if el not in self.__list_of_elements_1 + self.__list_of_elements_2 + list(elements_variables.keys()):
                elements_variables[el] = []
            for var in re.findall('[a-z'+''.join(self.__greek_letters)+']', amt):
                stoichiometry_variables[var] = []

        formula_structure = dict(
            formula_=init_formula,
            formula=formula,
            composition=composition,
            stoichiometry_vars=stoichiometry_variables,
            elements_vars=elements_variables,
            targets=[]
        )

        return formula_structure

    def __empty_structure(self):
        return dict(
            formula_='',
            formula='',
            composition=collections.defaultdict(str),
            stoichiometry_vars=collections.defaultdict(str),
            elements_vars=collections.defaultdict(str),
            targets=[]
        )

    def __is_correct_composition(self, formula, chem_compos):
        if chem_compos == {}:
            return False
        if any(el not in formula + 'M' or amt == '' for el, amt in chem_compos.items()):
            return False

        return True

    def get_mixture(self, material_name):

        mixture = {}

        for m in re.finditer('\(1-x\)(.*)-\({0,1}x\){0,1}(.*)', material_name.replace(' ', '')):
            mixture[m.group(1)] = {}
            mixture[m.group(2)] = {}
            mixture[m.group(1)]['fraction'] = '1-x'
            mixture[m.group(1)]['composition'] = self.get_structure_by_formula(m.group(1))['composition']
            mixture[m.group(2)]['fraction'] = 'x'
            mixture[m.group(2)]['composition'] = self.get_structure_by_formula(m.group(2))['composition']

            for i in [1, 2]:
                if m.group(i)[0] == '(' and m.group(i)[-1] == ')':
                    line = m.group(i)[1:-1]
                    parts = [s for s in re.split('[-+]{1}([\d\.]*[A-Z][^-+]*)', line) if s != '' and s != line]
                    for s in parts:
                        name = re.findall('([\d\.]*)([A-Z].*)', s.strip(' -+()'))[0]
                        mixture[name[1]] = {}
                        fraction = name[0]
                        if fraction == '': fraction = '1'
                        mixture[name[1]]['fraction'] = '('+fraction+')*'+mixture[m.group(i)]['fraction']
                        mixture[name[1]]['composition'] = self.get_structure_by_formula(name[1])['composition']
                    del mixture[m.group(i)]

        if mixture == {}:
            parts = [s for s in re.split('[-+]{1}([\d\.]*[A-Z][^-+]*)', material_name.replace(' ', '')) if s != '' and s != material_name.replace(' ', '')]
            for s in parts:
                name = re.findall('([\d\.]*)([A-Z].*)', s.strip(' -+()'))[0]
                mixture[name[1]] = {}
                fraction = name[0]
                if fraction == '': fraction = '1'
                mixture[name[1]]['fraction'] = fraction
                mixture[name[1]]['composition'] = self.get_structure_by_formula(name[1])['composition']

        for item in mixture:
            mixture[item]['fraction'] = self.__simplify(mixture[item]['fraction'])

        return mixture

    def get_chemical_structure(self, material_name):
        """
        The main function to obtain the closest chemical structure associated with a given material name
        :param material_name: string of material name
        :return: dictionary composition and stoichiometric variables
        """
        chemical_structure = dict(
            composition={},
            mixture={},
            fraction_vars={},
            elements_vars={},
            formula='',
            chemical_name=''
        )

        material_name = re.sub('[∙⋅](.*)', '', material_name)

        chemical_structure['mixture'] = self.get_mixture(material_name)

        chemical_structure['formula'] = ''.join(chemical_structure['mixture'].keys())
        if chemical_structure['formula'] == '':
            chemical_structure['formula'] = material_name

        # trying to extract chemical structure
        try:
            t_struct = self.get_structure_by_formula(chemical_structure['formula'])
        except:
            t_struct = self.__empty_structure()
            #print('Something went wrong!' + material_name)
            # return self.__empty_structure()

        chemical_structure['composition'] = t_struct['composition']
        # TODO: merge fraction variables

        # if material name is not proper formula look for it in DB (pubchem, ICSD)
        if not self.__is_correct_composition(chemical_structure['formula'], chemical_structure['composition']):
            # print('Looking in pubchem ' + material_name)
            chemical_structure['composition'] = collections.defaultdict(str)
            # chemical_structure['stoichiometry_vars'] = collections.defaultdict(str)
            chemical_structure['elements_vars'] = collections.defaultdict(str)

            pcp_compounds = pcp.get_compounds(material_name, 'name')
            if len(pcp_compounds) > 0:
                try:
                    chemical_structure['composition'] = self.get_structure_by_formula(pcp_compounds[0].molecular_formula)[
                        'composition']
                except:
                    chemical_structure['composition'] = collections.defaultdict(str)

        # if still cannot find composition look word by word <- this is a quick fix due to wrong tokenization,
        # will be removed probably
        if chemical_structure['composition'] == {}:
            # print('Looking part by part ' + material_name)
            for word in material_name.split(' '):
                try:
                    t_struct = self.get_structure_by_formula(word)
                except:
                    t_struct = self.__empty_structure()
                    #print('Something went wrong!' + material_name)
                if self.__is_correct_composition(word, t_struct['composition']):
                    chemical_structure['composition'] = t_struct['composition']

        chemical_structure['name'] = material_name

        return chemical_structure


class SimplifiedMaterialParser:
    def __init__(self):
        self.name = "ImprovedMaterialParser"

    def is_element(self, element):
        try:
            Element(element)
            return True
        except:
            return False

    def parse_formula(self, formula):

        try:
            composition = Composition(formula)

        except CompositionError:
            print("need a better parser!")

        if any([not self.is_element(key) for key in composition.keys()]):
            print("need to handle substitution")

        # if "(" and ")" in formula:
        # separate by elements and trailing expressions
        bits = re.findall('[A-Z][^A-Z]*', formula)
        parsed_formula = {}
        for bit in bits:
            if self.is_element(bit):
                parsed_formula[bit] = 1
            elif self.is_element(bit[0:2]):
                parsed_formula[bit[0:2]] = bit[2::]
            elif self.is_element(bit[0]):
                parsed_formula[bit[0]] = bit[1::]
            else:
                raise ValueError("Formula contains non-element.")

    # def is_chemical_formula(self, composition):
    #     current_element=''
    #     for letter in composition:
    #         if letter.isupper():
    #             current_element =


class TextParser:

    def __init__(self):
        self.name = "Parser"

    def extract_chemdata(self, text):
        doc = Document(text)
        cems = doc.cems
        chem_mentions = doc.records.serialize()
        materials = []
        for chem in chem_mentions:
            if 'names' in chem.keys():
                materials.append(chem["names"])
        return materials


def extract_materials(text):
    P = TextParser()
    M = MaterialParser()
    materials = P.extract_chemdata(text)
    extracted_materials = []
    for material_names in materials:
        newnames = []
        for name in material_names:
            print("Trying ", name)
            # handle ions
            if name[-1] in ['-', '+']:
                if name[-2] == " ":
                    name = name[0:-2]
                newnames.append(name)
                continue
            possible_material = M.make_pretty(M.parse_formula(name))
            if possible_material == '':
                possible_material = name
            #     print(possible_material)
            # except:
            #     possible_material = name
            #     print("still adding {}".format(possible_material))
            newnames.append(possible_material)
        newname = "{}{}".format(newnames[0], " " + str(newnames[1:] if len(newnames) > 1 else ""))
        extracted_materials.append(newname)
    print("extracted:", extracted_materials)
    return extracted_materials


def test_text_parsing():
    hard_text = """New TbFeAs(O,F) and DyFeAs(O,F) superconductors with critical temperatures Tc = 46 and 45 K and very 
    high critical fields, ≥100 T, have been prepared at 1100–1150 °C and 10–12 GPa, demonstrating that high pressure may 
    be used to synthesise late rare earth derivatives of the recently reported RFeAs(O,F) (R = La–Nd, Sm, Gd) high 
    temperature superconductors."""
    hard_text_materials = ["TbFeAsO", "TbFeAsF", "DyFeAsO", "DyFeAsF", "LaFeAsO", "CeFeAsO", "PrFeAsO", "NdFeAsO",
                           "SmFeAsO", "GdFeAsO", "LaFeAsF", "CeFeAsF", "PrFeAsF", "NdFeAsF", "SmFeAsF", "GdFeAsF"]

    easy_text = """From 1997 to present, continuous efforts have been made to 
    understand and improve the performance of LiFePO4. """
    easy_materials = "LiFePO4"

    P = TextParser()
    M = MaterialParser()
    # print(M.test_parsing(material_name="TbFeAsO", sentence="""New TbFeAs(O,F) and DyFeAs(O,F) superconductors with critical temperatures Tc = 46 and 45 K and very
    # high critical fields, ≥100 T, have been prepared at 1100–1150 °C and 10–12 GPa, demonstrating that high pressure may
    # be used to synthesise late rare earth derivatives of the recently reported RFeAs(O,F) (R = La–Nd, Sm, Gd) high
    # temperature superconductors."""))
    P.extract_chemdata(easy_text)
    P.extract_chemdata(hard_text)
    print(M.parse_formula("LiFePO4"))
    print(M.parse_formula(("LFP")))


#if __name__ == "__main__":
#    test_text_parsing()

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


def materials_extract(text):
    text_parser = TextParser()
    mat_parser = SimpleParser()
    extracted = text_parser.extract_chemdata(text)
    parsed = []
    missed = []
    for mentions in extracted:
        for mention in mentions:
            if mat_parser.parse(mention):
                parsed.append(mention)
            else:
                missed.append(mention)
    return parsed, missed
