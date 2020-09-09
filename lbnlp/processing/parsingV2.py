# coding=utf-8

__author__ = "Olga Kononova"
__maintainer__ = "Olga Kononova"
__email__ = "0lgaGkononova@yandex.ru"
__version__ = "6.0.0"

import os
import json
import regex as re
import collections
import sympy as smp
from sympy.abc import _clash
import pubchempy as pcp

from pprint import pprint


class MaterialParser:
    def __init__(self, verbose=False, pubchem_lookup=False, fails_log=False, dictionary_update=False):
        print("Initializing MaterialParser (forked) version 6.0.0")

        self.__filename = os.path.dirname(os.path.realpath(__file__))
        self.__pubchem_dictionary = json.loads(open(os.path.join(self.__filename, "resources/pubchem_dict.json")).read())
        self.__abbreviations = json.loads(open(os.path.join(self.__filename, "resources/abbreviations.json")).read())
        self.__ions = json.loads(open(os.path.join(self.__filename, "resources/ions_dictionary.json")).read())

        self.__element2name = self.__ions["elements"]
        self.__list_of_elements_1 = [el for el in self.__ions["elements"].keys() if len(el) == 1]
        self.__list_of_elements_2 = [el for el in self.__ions["elements"].keys() if len(el) == 2]
        self.__list_of_elements = self.__list_of_elements_1 + self.__list_of_elements_2
        self.__name2element = {v: k for k, v in self.__ions["elements"].items()}
        self.__anions = {ion["c_name"]: {"valency": ion["valency"], "e_name": ion["e_name"], "n_atoms": ion["n_atoms"]}
                         for ion in self.__ions["anions"]}
        self.__cations = {ion["c_name"]: {"valency": ion["valency"], "e_name": ion["e_name"], "n_atoms": ion["n_atoms"]}
                          for ion in self.__ions["cations"]}
        self.__list_of_anions = [ion["c_name"] for ion in self.__ions["anions"]]
        self.__list_of_cations = [ion["c_name"] for ion in self.__ions["cations"]]
        self.__diatomic_molecules = {"O2": collections.OrderedDict([("O", "2")]),
                                     "N2": collections.OrderedDict([("N", "2")]),
                                     "H2": collections.OrderedDict([("H", "2")]),
                                     "H2O": collections.OrderedDict([("H", "2"), ("O", "1")])}

        self.__greek_letters = [chr(i) for i in range(945, 970)]

        self.__prefixes2num = {"": 1, "mono": 1, "di": 2, "tri": 3, "tetra": 4, "pent": 5, "penta": 5, "hexa": 6,
                               "hepta": 7, "octa": 8, "nano": 9, "ennea": 9, "nona": 9, "deca": 10, "undeca": 11,
                               "dodeca": 12}
        self.__neg_prefixes = ["an", "de", "non"]

        self.__rome2num = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

        self.__fails_log = fails_log
        if fails_log:
            self.__pubchem_file = open(os.path.join(self.__filename, "pubchem_log"), "w")
            self.__pubchem_file.close()

        self.__pubchem = pubchem_lookup
        if pubchem_lookup:
            print("Pubchem lookup is on! Will search for unknown materials name in PubChem DB.")

        self.__dictionary_update = dictionary_update
        if dictionary_update:
            self.__dictionary_file = open(os.path.join(self.__filename, "dictionary_update"), "w")
            self.__dictionary_file.close()

        self.__verbose = verbose

    ###################################################################################################################
    # Parsing material name
    ###################################################################################################################

    def parse_material_string(self, material_string_):
        """
        Main method to convert chemical name into chemical formula and parse formula into chemical structure
        :param material_string_: < str> - material name and/or formula
        :return: dict(material_string: <str> - initial material string,
                      material_name: <str> - chemical name of material if found in the string
                      material_formula: <str> - chemical formula of material reconstructed from the string
                      composition: <list> of Object(dict):
                                formula: <str> - compound formula
                                amount: <str> - fraction of compound in mixture
                                elements: Object(dict) - {element: ratio/amount}
                      additives: <list> of <str> - list of dopped materials/elements found in material string
                      amount_vars: Object(dict) - {amount variable: <list> - values}
                      elements_vars: Object(dict) - {element variable: <list> - values}
                      oxygen_deficiency: <str> - sign of fraction (excess or deficiency) as given in material string
                      phase: <str> - material phase given in material string
                      is_acronym: <bool> - material string looks like acronym
        """

        material_string = self.cleanup_name(material_string_)
        if self.__verbose:
            print("After cleaning up string:", material_string_, "-->", material_string)

        if material_string in self.__list_of_elements:
            return self.__element_structure(material_string)

        if material_string in self.__diatomic_molecules:
            output_structure = self.__element_structure(material_string)
            output_structure["composition"][0]["elements"] = self.__diatomic_molecules[material_string]
            return output_structure

        """
        Converting material string into chemical formula
        """
        if material_string in self.__name2element:
            return self.__element_structure(self.__name2element[material_string])

        additives, material_string = self.separate_additives(material_string)
        if self.__verbose:
            print("After additives extraction:", material_string, "|", additives)

        _, material_formula, material_name = self.string2formula(material_string)


        """
        Extracting composition from chemical formula
        """
        material_compounds = self.split_formula_into_compounds(material_formula)

        if self.__verbose:
            print("\tAfter splitting:", material_formula, "-->", material_compounds)

        output_structure = self.__empty_structure().copy()
        output_structure['material_string'] = material_string_
        output_structure['material_name'] = material_name
        output_structure["material_formula"] = material_formula

        hydrate = [f for m, f in material_compounds if m == "H2O"]
        material_compounds = [(m, f) for m, f in material_compounds if m!= "H2O"]

        oxygen_deficiency = None
        for compound, amount in material_compounds:
            if compound in self.__abbreviations:
                if self.__verbose:
                    print("Found abbreviation:", compound, "-->", self.__abbreviations[compound])
                compound = self.__abbreviations[compound]
            composition = self.formula2composition(compound)
            output_structure["phase"] = composition["phase"]
            output_structure["amounts_vars"].update(composition["amounts_vars"])
            output_structure["elements_vars"].update(composition["elements_vars"])
            if composition["elements"] != {}:
                output_structure["composition"].append(
                    {"formula": composition["formula"],
                     "amount": amount,
                     "elements": composition["elements"]
                    }
                )
            if composition["oxygen_deficiency"]:
                oxygen_deficiency = composition["oxygen_deficiency"]
        output_structure["oxygen_deficiency"] = oxygen_deficiency

        """
        substituting additive into composition if it makes fractions to sum-up to integer
        adding to composition if it is compound
        """
        if len(additives) == 1:
            output_structure = self.substitute_additives(additives, output_structure)
        else:
            output_structure["additives"] = additives

        """
        appending hydrate part
        """
        if hydrate != []:
            output_structure["composition"].append(
                {"formula": "H2O",
                 "amount": hydrate[0],
                 "elements": self.__diatomic_molecules["H2O"]
                 }
            )

        """
        correction for negative stoichiometry and amount
        """
        try:
            if any(float(s) < 0.0 or float(s) > 100.0
                   for compound in output_structure["composition"] for e, s in compound["elements"].items()):
                output_structure["composition"] = []
        except:
            pass
        for compound in output_structure["composition"]:
            if len(re.findall("[b-mo-w]+", compound["amount"])) > 0:
                compound["amount"] = "1"

        """
        assigning abbreviations
        """
        output_structure["is_acronym"] = self.__is_acronym(output_structure)

        """
        finally, combine unified material formula from its compounds 
        """
        output_structure["material_formula"] = self.__combine_formula(output_structure["composition"])

        # if output_structure["composition"] == [] and self.__fails_log:
        #     with open(os.path.join(self.__filename, "fails_log"), "a") as f_log:
        #         f_log.write(material_name + "\n")

        return output_structure

    def formula2composition(self, formula):
        """
        Parsing chemical formula in composition
        :param formula: <str> chemical formula
        :return: dict(formula: <str> formula string corresponding to obtained composition
                     composition: <dict> element: fraction
                     fraction_vars: <dict> elements fraction variables: <list> values
                     elements_vars: <dict> elements variables: <list> values
                     hydrate: <str> if material is hydrate fraction of H2O
                     phase: <str> material phase appeared in formula
                    )
        """

        formula = formula.replace(" ", "")
        """
        check if there any phase specified
        """
        phase = ""
        if formula[0].islower():
            for m in re.finditer("([" + "".join(self.__greek_letters) + "]*)-{0,1}(.*)", formula):
                phase, formula = (m.group(1), m.group(2)) if m.group(2) != "" else ("", formula)
        """
        find oxygen deficiency
        """
        formula, oxygen_deficiency, oxygen_deficiency_sym = self.__get_oxygen_deficiency(formula)
        """
        converting fractions a(b+x)/c into (a/c*b+a/c*x)
        """
        formula = self.__fraction_convertion(formula)

        elements_variables = collections.defaultdict(str)
        stoichiometry_variables = collections.defaultdict(str)
        """
        check for any weird syntax (A,B)zElxEly...
        replacing with MzElxEly... and M = [A, B]
        """
        r = r"(\([A-Za-z\s]+[\/,\s]+[A-Za-z]+\))"
        for m in re.finditer(r, formula):
            elements_variables["M"] = re.split(r"[\/,]", m.group(0).strip('()'))
            formula = formula.replace(m.group(0), "M", 1)

        composition = self.__parse_formula(formula)

        if re.findall("[a-z]{4,}", formula) != [] and composition != {}:
            composition = collections.OrderedDict()
        """
        looking for variables in elements and stoichiometry 
        """
        for el, amt in composition.items():
            if el not in self.__list_of_elements + list(elements_variables.keys()) + ["□"]:
                elements_variables[el] = []
            for var in re.findall("[a-z" + "".join(self.__greek_letters) + "]", amt):
                stoichiometry_variables[var] = []

        rename_variables = [("R", "E"), ("A", "E"), ("T", "M")]
        for v1, v2 in rename_variables:
            if v1 in elements_variables and v2 in elements_variables and v1 + v2 in formula:
                elements_variables[v1 + v2] = []
                del elements_variables[v2]
                del elements_variables[v1]
                composition[v1 + v2] = composition[v2]
                del composition[v1]
                del composition[v2]

        if "M" in elements_variables and "e" in stoichiometry_variables:
            elements_variables["Me"] = []
            del elements_variables["M"]
            del stoichiometry_variables["e"]
            c = composition["M"][1:]
            composition["Me"] = c if c != "" else "1.0"
            del composition["M"]

        if not oxygen_deficiency and oxygen_deficiency_sym in stoichiometry_variables:
            oxygen_deficiency = None
        variables = [v for v in stoichiometry_variables.keys()
                     if [e for e, s in composition.items() if v in s] == ["O"]]
        oxygen_deficiency = variables[0] if len(variables) > 0 else oxygen_deficiency
        for var in variables:
            del stoichiometry_variables[var]
            composition["O"] = "1" if composition["O"] == var else composition["O"].replace(var, "").strip()
            formula = formula.replace(var, "")

        formula_structure = dict(elements=composition,
                                 amounts_vars={x: v for x, v in stoichiometry_variables.items()},
                                 elements_vars={e: v for e, v in elements_variables.items()},
                                 phase=phase,
                                 formula=formula,
                                 oxygen_deficiency=oxygen_deficiency)

        return formula_structure

    def __fraction_convertion(self, formula):
        formula_upd = formula
        r = r"([0-9\.]*)(\([0-9\.a-z]+[\-\+]+[0-9\.a-z]+\))(?=[/]*([0-9\.]*))"
        for m in re.finditer(r, formula_upd):
            a = m.group(1) if m.group(1) != '' else '1'
            c = m.group(3) if m.group(3) != '' else '1'
            bx = m.group(2).strip(')(')
            expr_str = a + '/' + c + '*' + bx[0] + bx[1] + a + '/' + c + '*' + bx[2]
            expr = str(smp.simplify(expr_str)).strip()
            if expr[0] == '-':
                s_expr = re.split(r"\+", expr)
                expr = s_expr[1] + s_expr[0]
            #expr_new = '(' + expr.strip() + ')'
            expr_new = expr.strip()
            expr_old = m.group(1) + m.group(2) + "/" + m.group(3) if m.group(3) != '' else m.group(1) + m.group(2)
            formula_upd = formula_upd.replace(expr_old, expr_new.strip(), 1)

        return re.sub(r"\s{1,}", "", formula_upd)

    def __get_oxygen_deficiency(self, formula):
        formula_upd = formula
        oxy_def = None
        oxy_def_sym = ""
        r = "".join([s for s in self.__greek_letters])
        r = "O[0-9]*([-+±∓]{1})[a-z" + r + "]{1}[0-9]*$"
        for m in re.finditer(r, formula_upd.rstrip(")")):
            end = formula_upd[m.start():m.end()]
            splt = re.split("[-+±]", end)
            oxy_def_sym = splt[-1]
            oxy_def = m.group(1)
            formula_upd = formula_upd[:m.start()] + formula_upd[m.start():].replace(end, splt[0])

        return formula_upd, oxy_def, oxy_def_sym

    def __parse_formula(self, init_formula):

        formula_dict = collections.OrderedDict()

        formula_dict = self.__parse_parentheses(init_formula, "1", formula_dict)
        """
        refinement of non-variable values
        """
        incorrect = []
        for el, amt in formula_dict.items():
            formula_dict[el] = self.__simplify(amt)
            if any(len(c) > 1 for c in re.findall("[A-Za-z]+", formula_dict[el])):
                incorrect.append(el)

        for el in incorrect:
            del formula_dict[el]

        return formula_dict

    def __parse_parentheses(self, init_formula, init_factor, curr_dict):
        r = r"\(((?>[^\(\)]+|(?R))*)\)\s*([-*\.\da-z\+/]*)"

        for m in re.finditer(r, init_formula):
            factor = "1"
            if m.group(2) != "":
                factor = m.group(2)

            factor = self.__simplify("(" + str(init_factor) + ")*(" + str(factor) + ")")
            unit_sym_dict = self.__parse_parentheses(m.group(1), factor, curr_dict)
            init_formula = init_formula.replace(m.group(0), "")

        unit_sym_dict = self.__get_sym_dict(init_formula, init_factor)
        for el, amt in unit_sym_dict.items():
            if el in curr_dict:
                if len(curr_dict[el]) == 0:
                    curr_dict[el] = amt
                else:
                    curr_dict[el] = "(" + str(curr_dict[el]) + ")" + "+" + "(" + str(amt) + ")"
            else:
                curr_dict[el] = amt

        return curr_dict

    def __get_sym_dict(self, f, factor):
        sym_dict = collections.OrderedDict()
        r = r"([A-Z□]{1}[a-z]{0,1})\s*([-\*\.\da-z" + "".join(self.__greek_letters) + r"\+\/]*)"

        def get_code_value(code, iterator):
            code_mapping = {"01": (iterator.group(1), iterator.group(2)),
                            "11": (iterator.group(1), iterator.group(2)),
                            "10": (iterator.group(1)[0], iterator.group(1)[1:] + iterator.group(2)),
                            "00": (iterator.group(1)[0], iterator.group(1)[1:] + iterator.group(2))}
            return code_mapping[code]

        for m in re.finditer(r, f):
            """
            checking for correct elements names
            """
            el_bin = "{0}{1}".format(str(int(m.group(1)[0] in self.__list_of_elements_1 + ["M", "□"])), str(
                int(m.group(1) in self.__list_of_elements + ["Ln", "M", "□"])))
            el, amt = get_code_value(el_bin, m)
            if amt.strip() == "":
                amt = "1"
            if el in sym_dict:
                sym_dict[el] = "(" + sym_dict[el] + ")" + "+" + "(" + amt + ")" + "*" + "(" + str(factor) + ")"
            else:
                sym_dict[el] = "(" + amt + ")" + "*" + "(" + str(factor) + ")"
            f = f.replace(m.group(), "", 1)
        if f.strip():
            return collections.OrderedDict()

        """
        refinement of non-variable values
        """
        for el, amt in sym_dict.items():
            sym_dict[el] = self.__simplify(amt)

        return sym_dict

    def __element_structure(self, element):
        return dict(
                material_string=element, material_name="", material_formula=element,
                additives=[], phase="", oxygen_deficiency=None,
                is_acronym=False,
                amounts_vars={}, elements_vars={},
                composition=[dict(
                    formula=element, amount="1", elements=collections.OrderedDict([(element, "1")])
                )]
            )

    ###################################################################################################################
    # Splitting list of materials
    ###################################################################################################################

    def is_materials_list(self, material_string):
        if (any(a + "s" in material_string.lower() for a in self.__anions.keys()) or "metal" in material_string) and \
                any(w in material_string for w in ["and ", ",", " of "]):
            return True

        return False

    def reconstruct_list_of_materials(self, material_string):
        """
        split material string into list of compounds when it"s given in form cation + several anions
        :param material_string: <str>
        :return: <list> of <str> chemical names
        """

        parts = [p for p in re.split(r"[\s\,]", material_string) if p != ""]

        anion = [(i, p[:-1]) for i, p in enumerate(parts) if
                 p[:-1].lower() in self.__anions.keys() or p[:-1].lower() == "metal"]
        cation = [(i, p) for i, p in enumerate(parts) if p.lower() in self.__cations.keys()
                  or p in self.__list_of_elements]
        valencies = [(i - 1, p.strip("()")) for i, p in enumerate(parts) if p.strip("()") in self.__rome2num and i != 0]

        result = []
        if len(anion) == 1:
            for c_i, c in cation:

                if c in self.__element2name:
                    name = [self.__element2name[c]]
                else:
                    name = [c.lower()]
                valency = "".join([v for v_i, v in valencies if v_i == c_i])
                if valency != "":
                    name.append("(" + valency + ")")
                name.append(anion[0][1])
                hydr_i = material_string.find("hydrate")
                if hydr_i > -1:
                    pref = []
                    while material_string[hydr_i - 1] != " " and hydr_i > 0:
                        pref.append(material_string[hydr_i - 1])
                        hydr_i -= 1

                    pref = "".join([p for p in reversed(pref)])

                    if pref not in self.__neg_prefixes:
                        name.append(pref + "hydrate")
                #result.append(("".join([n + " " for n in name]).strip(" "), valency))
                result.append(" ".join([n for n in name]))

        return result

    ###################################################################################################################
    # Reconstruct formula / dictionary lookup
    ###################################################################################################################

    def string2formula(self, material_string):

        if self.__verbose:
            print("Converting string to formula:")

        material_string_nodashes = ' '.join([t for t in re.split(r'\s|(?<=[a-z])-', material_string)])
        material_string_nodashes = material_string_nodashes[0].lower() + material_string_nodashes[1:]
        if material_string_nodashes in self.__pubchem_dictionary:
            return material_string, self.__pubchem_dictionary[material_string_nodashes], material_string_nodashes

        if material_string in self.__abbreviations:
            if self.__verbose:
                print("\tFound abbreviation:", material_string, "-->", self.__abbreviations[material_string])
            return material_string, self.__abbreviations[material_string], ""

        material_formula, material_name = self.extract_formula_from_string(material_string)
        if self.__verbose:
            print("\tAfter material name parsing into formula|name:\n",
                  material_string, "-->", material_formula, "|", material_name)

        if material_formula == "":
            material_formula = self.reconstruct_formula_from_string(material_name)
            if material_formula == "" and self.__pubchem:
                pcp_compounds = pcp.get_compounds(material_name, 'name')
                material_formula = pcp_compounds[0].molecular_formula if len(pcp_compounds) != 0 else ""
        if self.__verbose:
            print("\tAfter material formula reconstruction:\n",
                  material_string, "-->", material_formula, "|", material_name)

        material_name = "" if material_formula == "" else material_name
        material_formula = material_string if material_formula == "" else material_formula

        return material_string, material_formula, material_name

    def extract_formula_from_string(self, material_string):
        """
        Extracts chemical formula among chemical terms.
        This happens due to poor tokenization
        :param material_string
        :return: formula: <str> chemical formula found in material string
                 name: <str> chemical name found in material string
        """

        material_string = material_string.replace("[", " [").strip()
        material_string = re.sub(r"\s{2,}", " ", material_string)
        split = re.split(r"\s", material_string)

        if len(split) == 1:
            return "", material_string

        if len(split) == 2 and \
                all(t in self.__list_of_elements or t.rstrip('s').lower() in self.__list_of_anions+['metal'] for t in split):
            t1 = self.__element2name[split[0]] if split[0] in self.__element2name else split[0].rstrip('s').lower()
            t2 = self.__element2name[split[1]] if split[1] in self.__element2name else split[1].rstrip('s').lower()
            return "", t1 + " " + t2

        def try_formula(self, formula):
            if re.match(r"(\s*\([IV,]+\))", formula):
                return {}
            try:
                composition = self.formula2composition(formula)
            except:
                composition = {'elements': {}, 'elements_vars': {}}
            if self.__is_abbreviation_composition(composition):
                return {}
            return composition["elements"]

        formulas = []
        terms = []
        for part in split:
            composition_part = try_formula(self, part)
            if composition_part != {} or part.strip('+-1234567890') in self.__list_of_elements:
                formulas.append(part)
            else:
                terms.append(part)

        if len(formulas) > 1:
            #print("More than one formula")
            return material_string, ""

        if len(formulas) == 0:
            #print("No formula")
            return "", material_string
            #return "",  " ".join([t + " " for t in terms])

        if any(t.strip('+-1234567890') in self.__list_of_elements for t in terms):
            #print("Potentially many pieces of formula")
            return material_string, ""

        #print("One formula and chemical terms", terms)
        return formulas[0], " ".join([t for t in terms])

    def reconstruct_formula_from_string(self, material_name, valency=""):
        """
        reconstructing chemical formula for simple chemical names anion + cation
        :param material_name: <str> chemical name
        :param valency: <str> anion valency
        :return: <str> chemical formula
        """

        output_formula = ""
        material_name = re.sub(r"(\([IV]*\))", " \\1 ", material_name)
        material_name = re.sub(r"\s{2,}", " ", material_name)

        terms_list = []
        valency_list = []
        hydrate = ""
        cation_prefix_num = 0
        cation_data = {"c_name": "", "valency": [], "e_name": "", "n_atoms": 0}

        for t in material_name.split(" "):
            if t.strip("()") in self.__rome2num:
                valency_list.append(self.__rome2num[t.strip("()")])
                continue
            if "hydrate" in t.lower():
                hydrate = t
                continue
            terms_list.append(t.strip(" -"))

        terms_list_upd = []
        for t in terms_list:
            if all(p not in self.__prefixes2num for p in t.split("-")):
                terms_list_upd.extend(_ for _ in t.split("-"))
            else:
                terms_list_upd.append("".join([p for p in t.split("-")]))
        terms_list = terms_list_upd

        t = "".join([t + " " for t in terms_list]).lower().strip(" ")
        if t in self.__anions:
            return self.__anions[t]["e_name"]
        if t in self.__cations:
            return self.__cations[t]["e_name"]

        if len(terms_list) < 2:
            return output_formula

        anion = terms_list.pop().lower().rstrip("s")

        if valency == "":
            valency_num = max(valency_list + [0])
        else:
            valency_num = self.__rome2num[valency.strip("()")]

        next_term = terms_list.pop()
        if "hydrogen" in next_term.lower() and len(terms_list) != 0:
            anion = next_term + " " + anion
        else:
            terms_list += [next_term]

        _, anion_prefix_num, anion = ("", 0, anion) if anion.lower() in self.__anions else self.__get_prefix(anion)

        if anion in self.__anions:
            anion_data = self.__anions[anion].copy()
        elif anion in ["metal"] and terms_list[0].lower() in self.__cations:
            return self.__cations[terms_list[0].lower()]["e_name"]
        else:
            return output_formula

        if len(terms_list) >= 2:
            return output_formula

        if len(terms_list) == 1:
            cation = terms_list[0]
            _, cation_prefix_num, cation = ("", 0, cation) if cation.lower() in self.__cations \
                else self.__get_prefix(cation)
            if cation.lower() in self.__cations:
                cation = cation.lower()
                cation_data = self.__cations[cation].copy()
            elif cation in self.__element2name:
                cation_data = self.__cations[self.__element2name[cation]].copy()
            else:
                return output_formula

        if len(cation_data["valency"]) > 1 and valency_num != 0:
            if valency_num not in cation_data["valency"]:
                print("WARNING! Not common valency value for " + material_name)
                print(cation_data["valency"])
                print(valency_num)
            cation_data["valency"] = [valency_num]

        output_formula = self.__build_formula(anion=anion_data,
                                              cation=cation_data,
                                              cation_prefix_num=cation_prefix_num,
                                              anion_prefix_num=anion_prefix_num)

        if hydrate != "":
            _, hydrate_prefix_num, hydrate = self.__get_prefix(hydrate)
            hydrate_prefix = "" if hydrate_prefix_num in [0, 1] else str(hydrate_prefix_num)
            output_formula = output_formula + "·" + hydrate_prefix + "H2O"

        return output_formula

    def __build_formula(self, cation, anion, cation_prefix_num=0, anion_prefix_num=0):

        cation_stoich = 0
        anion_stoich = 0

        if anion_prefix_num + cation_prefix_num == 0 or anion_prefix_num * cation_prefix_num != 0:
            v_cation = abs(cation["valency"][0])
            v_anion = abs(anion["valency"][0])
            cm = self.__lcm(v_cation, v_anion)
            cation_stoich = cm // v_cation
            anion_stoich = cm // v_anion

        if anion_prefix_num != 0:
            anion_stoich = anion_prefix_num
            i = 0
            cation_stoich = 0
            while cation_stoich == 0 and i < len(cation["valency"]):
                cation_stoich = anion_prefix_num * abs(anion["valency"][0]) // abs(cation["valency"][i])
                i = i + 1

        if cation_prefix_num != 0:
            cation_stoich = cation_prefix_num
            anion_stoich = cation_prefix_num * abs(cation["valency"][0]) // abs(anion["valency"][0])

        anion_name_el = anion["e_name"]
        if anion_stoich > 1 and anion["n_atoms"] > 1:
            anion_name_el = "(" + anion_name_el + ")"

        cation_name_el = cation["e_name"]
        if cation_stoich > 1 and cation["n_atoms"] > 1:
            cation_name_el = "(" + cation_name_el + ")"

        return "{0}{1}{2}{3}".format(cation_name_el, self.__cast_stoichiometry(cation_stoich), anion_name_el,
                                     self.__cast_stoichiometry(anion_stoich))

    def __get_prefix(self, material_name):

        pref = ""
        pref_num = 0
        material_name_upd = material_name

        for p in self.__prefixes2num.keys():
            if material_name.lower().find(p) == 0 and p != "":
                pref = p
                pref_num = self.__prefixes2num[p]
                material_name_upd = material_name_upd[len(p):].strip("-")

                if material_name_upd == "xide":
                    material_name_upd = "oxide"

        return pref, pref_num, material_name_upd

    ###################################################################################################################
    # Splitting mixtures
    ###################################################################################################################

    def split_formula_into_compounds(self, material_name):
        """
        splitting mixture/composite/solid solution/alloy into compounds with fractions
        :param material_name: <str> material formula
        :return: <list> of <tuples>: (compound, fraction)
        """

        if self.__verbose:
            print("Splitting formula into compounds:")

        split = self.__split_formula(material_name)
        l = 0
        while len(split) != l:
            l = len(split)
            split = [p for s in split for p in self.__split_formula(s[0], s[1])]

        output = []
        for m, f in split:
            try:
                f = smp.simplify(f)
                if f.is_Number:
                    f = round(float(f), 3)
                f = str(f)
            except:
                f = "1"

            f = self.__simplify(f)
            output.append((m, f))

        output = [(self.__check_parentheses(m), f) for m, f in output]

        return output

    def __split_formula(self, material_name_, init_fraction="1"):

        re_str = r"(?<=[0-9\)])[-⋅·∙\∗](?=[\(0-9])|(?<=[A-Z])[-⋅·∙\∗](?=[\(0-9])|(?<=[A-Z\)])[-⋅·∙\∗](?=[A-Z])|(?<=[" \
                 r"0-9\)])[-⋅·∙\∗](?=[A-Z])"
        re_str = re_str + "".join(
            [r"|(?<=" + e + r")[-⋅·∙\∗](?=[\(0-9A-Z])" for e in self.__list_of_elements])
        re_str = re_str + r"|[-·]([nx0-9\.]H2O)"

        material_name = material_name_.replace(" ", "")

        if "(1-x)" == material_name[0:5] or "(100-x)" == material_name[0:7] :
            material_name = material_name.replace("(x)", "x")
            parts = re.findall(r"\(10{0,2}-x\)(.*)[-+·∙\∗⋅]x(.*)", material_name)
            parts = parts[0] if parts != [] else (material_name[5:], "")
            return [(parts[0].lstrip(" ·*⋅"), "1-x"), (parts[1].lstrip(" ·*"), "x")]

        parts = [p for p in re.split(re_str, material_name) if p]

        if len(parts) > 1:
            parts_upd = [p for part in parts for p in
                         re.split(r"(?<=[A-Z\)])[-·∙\∗⋅](?=[xyz])|(?<=O[0-9\)]+)[-·∙\∗⋅](?=[xyz])", part)]
        else:
            parts_upd = parts

        if any(m.strip("0987654321") in self.__list_of_elements for m in parts_upd[:-1]):
            parts_upd = ["".join([p + "-" for p in parts_upd]).rstrip("-")]

        merged_parts = [parts_upd[0]]
        for m in parts_upd[1:]:
            if re.findall("[A-Z]", m) == ["O"]:
                to_merge = merged_parts.pop() + "-" + m
                merged_parts.append(to_merge)
            else:
                merged_parts.append(m)

        composition = []
        for m in merged_parts:
            fraction = ""
            i = 0
            while i < len(m) and not m[i].isupper():
                fraction = fraction + m[i]
                i += 1
            fraction = fraction.strip("()")
            if fraction == "":
                fraction = "1"
            else:
                m = m[i:]

            fraction = "(" + fraction + ")*(" + init_fraction + ")"

            if m != "":
                composition.append((m, fraction))
        return composition

    def separate_additives(self, material_name):
        """
        resolving doped part in material string
        :param material_name: <str> material string
        :return: <list> of additives,
                <str> new material name
        """
        new_material_name = material_name
        additives = []

        new_material_name = new_material_name.replace("codoped", "doped")
        new_material_name = new_material_name.replace("co-doped", "doped")

        # checking for "doped with"
        for r in ["activated", "modified", "stabilized", "doped", "added"]:
            parts = [w for w in re.split(r + " with", new_material_name) if w != ""]
            if len(parts) > 1:
                new_material_name = parts[0].strip(" -+")
                additives.append(parts[1].strip())

        # checking for element-doped prefix
        for r in ["activated", "modified", "stabilized", "doped", "added"]:
            parts = [w for w in re.split(r"(.*)[-\s]{1}" + r + " (.*)", new_material_name) if w != ""]
            if len(parts) > 1:
                new_material_name = parts.pop()
                additives.extend(p for p in parts)

        if "%" in new_material_name:
            new_material_name = new_material_name.replace(".%", "%")
            parts = re.split(r"[\-+:·]{0,1}\s*[0-9x\.]*\s*[vmolwt\s]*\%", new_material_name)

        if len(parts) > 1:
            new_material_name = parts[0].strip(" -+")
            additives.extend(d.strip() for d in parts[1:] if d != "")

        for part_ in new_material_name.split(":"):
            part_ = part_.strip()

            part = part_
            if any(e in part for e in self.__list_of_elements_2):
                for e in self.__list_of_elements_2:
                    part = part.replace(e, "&&")

            if all(e.strip("zyx,+0987654321. ") in self.__list_of_elements_1 + ["R"] + ["&&"]
                   for e in re.split(r"[\s,/]", part) if e != ""):
                additives.append(part_.strip(" "))
            else:
                new_material_name = part_.strip(" ")

        additives_final = [a.strip() for s in additives for a in re.split(r"[\s,\-/]|and", s) if a.strip() != ""]

        return additives_final, new_material_name.rstrip("( ,.:;-±/+")

    def substitute_additives(self, additives, material_structure):
        """
        analyzes additives and adjust them to the composition
        :param additives: list of additives
        :param material_structure: parsed material structure
        :return: updated material structure
        """
        additive = additives[0]

        material_structure_new = material_structure.copy()
        try:
            additive_composition = self.formula2composition(additive)
        except:
            additive_composition = self.__empty_composition().copy()
        if len(additive_composition["elements"]) > 1:
            #print('-->', "Additive is compound")
            for structure in material_structure_new["composition"]:
                structure["amount"] = structure["amount"] + "-x"
            material_structure_new["composition"].append(
                {"formula": additive,
                 "amount": "x",
                 "elements": additive_composition["elements"]
                 }
            )
        elif all(c["elements"] != {} for c in material_structure_new["composition"]):
            #print('-->', "Additive is element with fraction")
            formula, composition = self.__substitute_additive(additive, material_structure_new["material_formula"],
                                                              material_structure_new["composition"])
            if formula != material_structure_new["material_formula"]:
                material_structure_new["additives"] = []
                material_structure_new["material_formula"] = formula
                material_structure_new["composition"] = composition
            else:
                material_structure_new["additives"] = additives
        else:
            #print('-->', "Additive is something else")
            material_structure_new["additives"] = additives

        return material_structure_new

    def __substitute_additive(self, additive, material_formula, material_composition):

        new_material_composition = []
        new_material_formula = material_formula

        if additive[-1] == "+":
            additive = additive.rstrip("+0987654321")

        r = r"^[x0-9\.]+|[x0-9\.]+$"
        coeff = re.findall(r, additive)
        element = [s for s in re.split(r, additive) if s != ""][0]

        if coeff == [] or element not in self.__list_of_elements:
            return new_material_formula, material_composition

        for compound in material_composition:
            expr = "".join(["(" + v + ")+" for e, v in compound["elements"].items()]).rstrip("+")

            coeff = coeff[0] if not re.match("^[0]+[1-9]", coeff[0]) else "0." + coeff[0][1:]
            expr = expr + "+(" + coeff + ")"

            if self.__is_int(self.__simplify(expr)):
                new_name = element + coeff + compound["formula"]
                new_composition = compound["elements"].copy()
                new_composition.update({element: coeff})
                new_composition.move_to_end(element, last=False)

                new_material_composition.append(dict(
                    formula=new_name,
                    amount=compound["amount"],
                    elements=new_composition
                ))
                new_material_formula = new_material_formula.replace(compound["formula"], new_name)
            else:
                new_material_composition.append(dict(
                    formula=compound["formula"],
                    amount=compound["amount"],
                    elements=compound["elements"]
                ))

        return new_material_formula, new_material_composition

    ###################################################################################################################
    # Resolving abbreviations
    ###################################################################################################################

    def __is_abbreviation(self, word):
        if all(c.isupper() for c in re.sub(r"[0-9x\-\(\)\.]", "", word)) and len(re.findall("[A-NP-Z]", word)) > 1:
            return True

        return False

    def build_abbreviations_dict(self, materials_list, paragraph):
        """
        constructing dictionary of abbreviations appeared in material list
        :param paragraph: <list> of <str> list of sentences to look for abbreviations names
        :param materials_list: <list> of <str> list of materials entities
        :return: <dict> abbreviation: corresponding string
        """

        abbreviations_dict = {t: "" for t in materials_list if self.__is_abbreviation(t.replace(" ", "")) and t != ""}
        not_abbreviations = list(set(materials_list) - set(abbreviations_dict.keys()))

        # first find abreviations in current materials list
        for abbr in abbreviations_dict.keys():

            for material_name in not_abbreviations:
                if sorted(re.findall("[A-NP-Z]", abbr)) == sorted(re.findall("[A-NP-Z]", material_name)):
                    abbreviations_dict[abbr] = material_name

        # for all other abbreviations going through the paper text
        for abbr, name in abbreviations_dict.items():
            sents = " ".join([s + " " for s in paragraph if abbr in s]).strip(" ").split(abbr)
            i = 0
            while abbreviations_dict[abbr] == "" and i < len(sents):
                sent = sents[i]
                for tok in sent.split(" "):
                    if sorted(re.findall("[A-NP-Z]", tok)) == sorted(re.findall("[A-NP-Z]", abbr)):
                        abbreviations_dict[abbr] = tok
                i += 1

        for abbr in abbreviations_dict.keys():
            parts = re.split("-", abbr)
            if all(p in abbreviations_dict for p in parts) and abbreviations_dict[abbr] == "" and len(parts) > 1:
                name = "".join("(" + abbreviations_dict[p] + ")" + "-" for p in parts).rstrip("-")
                abbreviations_dict[abbr] = name

        empty_list = [abbr for abbr, name in abbreviations_dict.items() if name == ""]
        for abbr in empty_list:
            del abbreviations_dict[abbr]

        return abbreviations_dict

    ###################################################################################################################
    # Methods to substitute variables
    ###################################################################################################################

    def __get_values(self, string, mode):
        values = []
        max_value = None
        min_value = None

        if len(string) == 0:
            return dict(values=[], max_value=None, min_value=None)

        # given range
        if mode == "range":
            min_value, max_value = string[0]
            max_value = max_value.rstrip("., ")
            min_value = min_value.rstrip("., ")
            max_value = re.sub("[a-z]*", "", max_value)
            min_value = re.sub("[a-z]*", "", min_value)
            try:
                max_value = round(float(smp.simplify(max_value)), 4)
                min_value = round(float(smp.simplify(min_value)), 4) if min_value != "" else None
            except Exception as ex:
                max_value = None
                min_value = None
                template = "An exception of type {0} occurred when use sympy. Arguments:\n{1!r}."
                message = template.format(type(ex).__name__, ex.args)
                print(message)
            values = []

            return dict(values=values, max_value=max_value, min_value=min_value)

        # given list
        if mode == "values":
            values = re.split(r"[,\s]", re.sub("[a-z]+", "", string[0]))
            try:
                values = [round(float(smp.simplify(c.rstrip("., "))), 4) for c in values if
                          c.rstrip("., ") not in ["", "and"]]
                max_value = max(values) if values != [] else None
                min_value = min(values) if len(values) > 1 else None
            except Exception as ex:
                values = []
                max_value = None
                min_value = None
                template = "An exception of type {0} occurred when use sympy. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)

        return dict(values=values, max_value=max_value, min_value=min_value)

    def get_stoichiometric_values(self, var, sentence):
        """
        find numeric values of var in sentence
        :param var: <str> variable name
        :param sentence: <str> sentence to look for
        :return: <dict>: max_value: upper limit
                        min_value: lower limit
                        values: <list> of <float> numeric values
        """
        values = dict(values=[], max_value=None, min_value=None)

        regs = [(var + r"\s*=\s*([-]{0,1}[0-9\.\,/and\s]+)[\s\)\]\,]", "values"),
                (var + r"\s*=\s*([0-9\.]+)\s*[-–]\s*([0-9\.\s]+)[\s\)\]\,m\%]", "range"),
                (r"([0-9\.\s]*)\s*[<≤⩽]{0,1}\s*" + var + r"\s*[<≤⩽>]{1}\s*([0-9\.\s]+)[\s\)\]\.\,]", "range"),
                (var + r"[a-z\s]*from\s([0-9\./]+)\sto\s([0-9\./]+)", "range"),
                ]

        for r, m in regs:
            if values["values"] == [] and values["max_value"] is None:
                r_res = re.findall(r, sentence.replace(" - ", "-"))
                values = self.__get_values(r_res, m)

        return values

    def get_elements_values(self, var, sentence):
        """
        find elements values for var in the sentence
        :param var: <str> variable name
        :param sentence: <str> sentence to look for
        :return: <list> of <str> found values
        """
        values = re.findall(var + r"\s*[=:]{1}\s*([A-Za-z0-9\+,\s]+)", sentence)
        values = [c.rstrip("0987654321+") for v in values for c in re.split(r"[,\s]", v)
                  if c.rstrip("0987654321+") in self.__list_of_elements]

        return list(set(values))

    ###################################################################################################################
    # Misc
    ###################################################################################################################

    def cleanup_name(self, material_name):
        """
        cleaning up material name - fix due to tokenization imperfectness - TO BE REMOVED FROM FINAL VERSION
        :param material_name: <str> material string
        :return: <str> updated material string
        """
        #TODO: [Ti(N3)6]2−, Cu(NO3)2⋅4 H2O

        # correct dashes
        dashes = [45, 173, 8722, ord("\ue5f8")] + [i for i in range(8208, 8214)]
        re_str = "".join([chr(c) for c in dashes])
        re_str = "\s*[" + re_str + "]\s*"
        material_name = re.sub(re_str, chr(45), material_name)
        material_name = re.sub(r"\s*\+\s*", "+", material_name)

        material_name = material_name.replace(chr(160), "")

        # correcting dots
        dots = [42, 215, 8226, 8270, 8729, 8901, 215, 65106, 65381, 12539, 9072]
        re_str = "".join([chr(c) for c in dots])
        re_str = "[\\" + re_str + "]"
        material_name = re.sub(re_str, chr(183), material_name)
        material_name = re.sub(r"\s"+chr(183)+r"\s", chr(183), material_name)

        # correcting slashes
        slashes = [8725]
        re_str = "".join([chr(c) for c in slashes])
        re_str = "[\\" + re_str + "]"
        material_name = re.sub(re_str, chr(47), material_name)

        material_name = re.sub(r"\s*([-+±]){1}\s*([" + "".join([c for c in self.__greek_letters]) + "]{1})", "\\1δ",
                               material_name)

        # removing phase
        for c in ["(s)", "(l)", "(g)", "(aq)"]:
            material_name = material_name.replace(c, "")

        #removing trach words
        trash_list = ["powder", "ceramic", "rear", "earth", "micro", "nano", "coat", "crystal", "particl", "glass"]
        for word in trash_list:
            material_name = re.sub("[A-Za-z-]*" + word + "[A-Za-z-]*", "", material_name)
            material_name = re.sub(word.capitalize() + "[a-z-]*", "", material_name)

        if any(a in material_name for a in ["→", "⟶", "↑", "↓", "↔", "⇌", "⇒", "⇔", "⟹"]):
            return ""

        if "hbox" in material_name.lower():
            material_name = re.sub(r"(\\\\[a-z\(\)]+)", "", material_name)
            for t in ["{", "}", "_", " "]:
                material_name = material_name.replace(t, "")
            material_name = material_name.rstrip("\\")

        material_name = re.sub(r"\({0,1}[0-9\.]*\s*[⩽≤<]{0,1}\s*[x,y]{0,1}\s*[⩽=≤<]\s*[0-9\.-]*\){0,1}", "",
                               material_name)

        if material_name == "" or len([c for c in material_name if c.isalpha()]) < 1:
            return ""

        for c in [r"\(⩾99", r"\(99", r"\(98", r"\(90", r"\(95", r"\(96", r"\(Alfa", r"\(Aldrich", r"\(A.R.",
                  r"\(Aladdin", r"\(Sigma", r"\(A.G", r"\(Fuchen", r"\(Furuuchi", r"\(AR\)", "（x", r"\(x", r"\(Acr[a-z]*",
                  r"\(Koj", r"\(Sho", r"\(＞99"]:
            split = re.split(c, material_name)
            if len(split) > 1 and (len(split[-1]) == "" or all(not s.isalpha() for s in split[-1])):
                material_name = "".join([s for s in split[:-1]])

        replace_dict = {"oxyde": "oxide",
                        "luminum": "luminium",
                        "magneshium": "magnesium",
                        "stanate": "stannate",
                        "sulph": "sulf",
                        "buter": "butyr",
                        "butir": "butyr",
                        "butly": "butyl",
                        "ethly": "ethyl",
                        "ehtyl": "ethyl",
                        "Abstract ": "",
                        "phio": "thio",
                        "uim": "ium",
                        "butryal": "butyral",
                        "ooper": "opper",
                        "acac": "CH3COCHCOCH3",
                        "glasses": "",
                        "glass": "",
                        "ceramics": "",
                        "europeam": "europium",
                        "siliminite": "sillimanite",
                        "acethylene": "acetylene",
                        "iso-pro": "isopro",
                        "anhydrous": "",
                        "lathanum": "lanthanum",
                        "bulk": "",
                        "Bulk": "",
                        "()": "",
                        "uium": "ium",
                        "Anhydrous": "",
                        "sodiam": "sodium"
                        }

        for typo, correct in replace_dict.items():
            material_name = material_name.replace(typo, correct)

        if material_name[-2:] == "/C":
            material_name = material_name[:-2]

        material_name = re.sub(r"(poly[\s-])(?=[a-z])", "poly", material_name)

        if material_name[-2:] == "/C":
            material_name = material_name[:-2]

        if len(material_name.split(" ")) > 1:
            for v in re.findall(r"[a-z](\([IV,]+\))", material_name):
                material_name = material_name.replace(v, " " + v)

        if material_name != "":
            material_name = self.__check_parentheses(material_name)

        trash_symbs = ["#", "$", "!", "@", "©", "®", chr(8201), "Ⓡ", "\u200b"]
        for c in trash_symbs:
            material_name = material_name.replace(c, "")

        material_name = material_name.replace("[", "(")
        material_name = material_name.replace("]", ")")
        material_name = material_name.replace("{", "(")
        material_name = material_name.replace("}", "(")

        material_name = material_name.lstrip(") -")
        material_name = material_name.rstrip("( ,.:;-±/∓")

        if len(material_name) == 1 and material_name not in self.__list_of_elements_1:
            return ""

        if len(material_name) == 2 and \
                material_name not in self.__list_of_elements_2 and \
                material_name.rstrip("234") not in self.__list_of_elements_1 and \
                any(c not in self.__list_of_elements_1 for c in material_name):
            return ""

        material_name = re.sub(r"\s([0-9\.]*H2O)$", chr(183) + "\\1", material_name)

        material_name = self.combine_formula_parts(material_name)
        material_name = self.__check_parentheses(material_name)

        return material_name.strip()

    def check_parentheses(self, formula):
        return self.__check_parentheses(formula)

    def parentheses_balanced(self, formula):
        opening_par = []

        for i, c in enumerate(formula):
            if c == '(':
                opening_par.append(i)
            if c == ')':
                if opening_par == []:
                    return False
                else:
                    opening_par = opening_par[:-1]

        return opening_par == []

    def combine_formula_parts(self, formula):
        """
        removes extra spaces if they break formula
        :param formula:
        :return:
        """
        parts = [p for p in reversed(re.split(r"\s", formula))]
        formula_upd = ""
        while parts:
            part = parts.pop()
            if self.parentheses_balanced(part):
                formula_upd = formula_upd + part + " "
            else:
                part_i = part
                while parts and not self.parentheses_balanced(part_i):
                    part = parts.pop()
                    part_i = part_i + part
                formula_upd = formula_upd + part_i + " "

        formula_upd = formula if formula_upd == "" else formula_upd.strip()

        return formula_upd


    def __check_parentheses(self, formula):
        """
        :param formula:
        :return:
        """

        new_formula = formula
        new_formula = new_formula.replace("{", "(")
        new_formula = new_formula.replace("}", ")")
        new_formula = new_formula.replace("[", "(")
        new_formula = new_formula.replace("]", ")")

        if new_formula[0] == '(' and new_formula[-1] == ')' and self.parentheses_balanced(new_formula[1:-1]):
            new_formula = new_formula[1:-1]

        if new_formula[0] == '(' and self.parentheses_balanced(new_formula[1:]):
            new_formula = new_formula[1:]

        if new_formula[-1] == ')' and self.parentheses_balanced(new_formula[:-1]):
            new_formula = new_formula[:-1]

        if self.parentheses_balanced(new_formula):
            return new_formula

        par_open = []
        par_close = []
        for i, c in enumerate(new_formula):
            if c == '(':
                par_open.append(i)
            if c == ')':
                if par_open == []:
                    par_close.append(i)
                else:
                    par_open = par_open[:-1]

        for i in par_close:
            new_formula = '(' + new_formula
        for i in par_open:
            new_formula = new_formula + ')'

        return new_formula

    def __simplify(self, value):
        """
        simplifying stoichiometric expression
        :param value: string
        :return: string
        """
        for l in self.__greek_letters:
            _clash[l] = smp.Symbol(l)

        new_value = value
        for i, m in enumerate(re.finditer(r"(?<=[0-9])([a-z" + "".join(self.__greek_letters) + "])", new_value)):
            new_value = new_value[0:m.start(1) + i] + "*" + new_value[m.start(1) + i:]
        new_value = smp.simplify(smp.sympify(new_value, _clash))
        if new_value.is_Number:
            new_value = round(float(new_value), 3)
        else:
            new_value = new_value.evalf(3)
        new_value = re.sub('\.0+(?![0-9])', '', str(new_value).replace(" ", ""))
        if new_value[0] == "-":
            split = re.split("\+", new_value)
            new_value = split[1] + split[0] if len(split) == 2 else new_value

        return new_value

    def __lcm(self, x, y):
        """This function takes two
        integers and returns the L.C.M."""

        # choose the greater number
        lcm = None
        if x > y:
            greater = x
        else:
            greater = y

        found = False
        while not found:
            if (greater % x == 0) and (greater % y == 0):
                lcm = greater
                found = True
            greater += 1

        return lcm

    def __cast_stoichiometry(self, value):

        value = float(value)
        if value == 1.0:
            return ""
        if value * 1000 % 1000 == 0.0:
            return str(int(value))

        return str(value)

    def __empty_composition(self):
        return {"formula": "",
                "elements": collections.OrderedDict(),
                "amounts_vars": {}, "elements_vars": {},
                "phase": None, "oxygen_deficiency": None}

    def __empty_structure(self):
        return {"material_string": "", "material_name": "", "material_formula": "",
                "phase": None, "additives": [], "oxygen_deficiency": None,
                "is_acronym": False,
                "amounts_vars": {}, "elements_vars": {},
                "composition": []}

    def __is_int(self, num):
        try:
            return round(float(num), 3) == round(float(num), 0)
        except:
            return False

    def __is_abbreviation_composition(self, composition):
        if all(e.isupper() and s in ["1.0", "1", "s"] for e, s in composition["elements"].items()):
            return True

        elements_vars = [el for el in composition["elements_vars"].keys() if len(el) == 1 and el.isupper()]
        if len(elements_vars) > 1:
            return True

        return False


    def __is_acronym(self, structure):
        if any(all(e.isupper() and s in ["1.0", "1"] for e, s in compound["elements"].items()) for compound in
               structure["composition"]):
            return True

        elements_vars = [el for el in structure["elements_vars"].keys() if len(el) == 1 and el.isupper()]
        if len(elements_vars) > 1:
            return True

        if all(c.isupper() for c in structure["material_formula"]) and any(
                c not in self.__list_of_elements_1 for c in structure["material_formula"]):
            return True

        if re.findall("[A-Z]{3,}", structure["material_formula"]) != [] and \
                all(w not in structure["material_formula"] for w in ["CH", "COO", "OH"]):
            return True

        if "PV" == structure["material_formula"][0:2]:
            return True

        return False

    def get_element(self, name):
        if name in self.__anions:
            return self.__anions[name]["e_name"]
        if name in self.__cations:
            return self.__cations[name]["e_name"]
        return ""

    def __combine_formula(self, material_composition):
        formula = ""
        if len(material_composition) == 1:
            return material_composition[0]["formula"].replace("*", "")

        coeff = ""
        for c in sorted(material_composition):
            if coeff != "":
                coeff = self.__simplify(coeff)
            if all(ch.isdigit() or ch == "." for ch in c["amount"]):
                coeff = self.__cast_stoichiometry(c["amount"])
            else:
                coeff = "(" + c["amount"] + ")" if c["amount"] != "x" else c["amount"]

            sign = "-"
            if "H2O" in c["formula"]:
                sign = "·"

            formula = formula + sign + coeff + c["formula"]

        formula = formula.replace("*", "")

        return formula.lstrip("-")