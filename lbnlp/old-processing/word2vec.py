from matscholar_core.database import AtlasConnection
from chemdataextractor.doc import Paragraph
from gensim.utils import deaccent
from matscholar_core.processing import parsing
from pymatgen.core.composition import Composition
from monty.fractions import gcd_float
from tqdm import tqdm
import regex
import string
import pickle


class DataPreparation:
    RAW_ABSTRACT_COL = "abstracts_w2v"
    TOK_ABSTRACT_COL = "abstracts_w2v_tokens"
    TTL_FILED = "title"
    ABS_FIELD = "abstract"
    DOI_FIELD = "doi"

    UNITS = ['K', 'h', 'V', 'wt', 'wt.', 'MHz', 'kHz', 'GHz', 'Hz', 'days', 'weeks',
             'hours', 'minutes', 'seconds', 'T', 'MPa', 'GPa', 'at.', 'mol.',
             'at', 'm', 'N', 's-1', 'vol.', 'vol', 'eV', 'A', 'atm', 'bar',
             'kOe', 'Oe', 'h.', 'mWcm−2', 'keV', 'MeV', 'meV', 'day', 'week', 'hour',
             'minute', 'month', 'months', 'year', 'cycles', 'years', 'fs', 'ns',
             'ps', 'rpm', 'g', 'mg', 'mAcm−2', 'mA', 'mK', 'mT', 's-1', 'dB',
             'Ag-1', 'mAg-1', 'mAg−1', 'mAg', 'mAh', 'mAhg−1', 'm-2', 'mJ', 'kJ',
             'm2g−1', 'THz', 'KHz', 'kJmol−1', 'Torr', 'gL-1', 'Vcm−1', 'mVs−1',
             'J', 'GJ', 'mTorr', 'bar', 'cm2', 'mbar', 'kbar', 'mmol', 'mol', 'molL−1',
             'MΩ', 'Ω', 'kΩ', 'mΩ', 'mgL−1', 'moldm−3', 'm2', 'm3', 'cm-1', 'cm',
             'Scm−1', 'Acm−1', 'eV−1cm−2', 'cm-2', 'sccm', 'cm−2eV−1', 'cm−3eV−1',
             'kA', 's−1', 'emu', 'L', 'cmHz1', 'gmol−1', 'kVcm−1', 'MPam1',
             'cm2V−1s−1', 'Acm−2', 'cm−2s−1', 'MV', 'ionscm−2', 'Jcm−2', 'ncm−2',
             'Jcm−2', 'Wcm−2', 'GWcm−2', 'Acm−2K−2', 'gcm−3', 'cm3g−1', 'mgl−1',
             'mgml−1', 'mgcm−2', 'mΩcm', 'cm−2s−1', 'cm−2', 'ions', 'moll−1',
             'nmol', 'psi', 'mol·L−1', 'Jkg−1K−1', 'km', 'Wm−2', 'mass', 'mmHg',
             'mmmin−1', 'GeV', 'm−2', 'm−2s−1', 'Kmin−1', 'gL−1', 'ng', 'hr', 'w',
             'mN', 'kN', 'Mrad', 'rad', 'arcsec', 'Ag−1', 'dpa', 'cdm−2',
             'cd', 'mcd', 'mHz', 'm−3', 'ppm', 'phr', 'mL', 'ML', 'mlmin−1', 'MWm−2',
             'Wm−1K−1', 'Wm−1K−1', 'kWh', 'Wkg−1', 'Jm−3', 'm-3', 'gl−1', 'A−1',
             'Ks−1', 'mgdm−3', 'mms−1', 'ks', 'appm', 'ºC', 'HV', 'kDa', 'Da', 'kG',
             'kGy', 'MGy', 'Gy', 'mGy', 'Gbps', 'μB', 'μL', 'μF', 'nF', 'pF', 'mF',
             'A', 'Å', 'A˚', "μgL−1"]

    ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
                'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
                'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']
    ELEMENT_NAMES = ['hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen', 'oxygen', 'fluorine',
                     'neon', 'sodium', 'magnesium', 'aluminium', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon',
                     'potassium', 'calcium', 'scandium', 'titanium', 'vanadium', 'chromium', 'manganese', 'iron',
                     'cobalt', 'nickel', 'copper', 'zinc', 'gallium', 'germanium', 'arsenic', 'selenium', 'bromine',
                     'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium', 'niobium', 'molybdenum', 'technetium',
                     'ruthenium', 'rhodium', 'palladium', 'silver', 'cadmium', 'indium', 'tin', 'antimony', 'tellurium',
                     'iodine', 'xenon', 'cesium', 'barium', 'lanthanum', 'cerium', 'praseodymium', 'neodymium',
                     'promethium', 'samarium', 'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium',
                     'thulium', 'ytterbium', 'lutetium', 'hafnium', 'tantalum', 'tungsten', 'rhenium', 'osmium',
                     'iridium', 'platinum', 'gold', 'mercury', 'thallium', 'lead', 'bismuth', 'polonium', 'astatine',
                     'radon', 'francium', 'radium', 'actinium', 'thorium', 'protactinium', 'uranium', 'neptunium',
                     'plutonium', 'americium', 'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
                     'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium', 'dubnium', 'seaborgium', 'bohrium',
                     'hassium', 'meitnerium', 'darmstadtium', 'roentgenium', 'copernicium', 'nihonium', 'flerovium',
                     'moscovium', 'livermorium', 'tennessine', 'oganesson', 'ununennium']
    ELEMENTS_AND_NAMES = ELEMENTS + ELEMENT_NAMES + [en.capitalize() for en in ELEMENT_NAMES]

    NR_BASIC = regex.compile(r'^[+-]?\d*\.?\d+\(?\d*\)?+$', regex.DOTALL)
    NR_AND_UNIT = regex.compile(r'^([+-]?\d*\.?\d+\(?\d*\)?+)([\p{script=Latin}|Ω|μ]+.*)', regex.DOTALL)

    # elemement with the valence state in parenthesis
    ELEMENT_VALENCE_IN_PAR = regex.compile(r'^('+'|'.join(ELEMENTS_AND_NAMES) +
                                           ')(\(([IV|iv]|[Vv]?[Ii]{0,3})\))$')
    ELEMENT_DIRECTION_IN_PAR = regex.compile(r'^('+'|'.join(ELEMENTS_AND_NAMES) + ')(\(\d\d\d\d?\))')

    # exactly IV, VI or has 2 consecutive II, or roman in parenthesis: is not a simple formula
    VALENCE_INFO = regex.compile(r'(II+|^IV$|^VI$|\(IV\)|\(V?I{0,3}\))')

    PUNCT = list(string.punctuation) + ['"', '“', '”', '≥', '≤', '×']

    # ROMAN_NR_PR = regex.compile(r'\((IV|V?I{0,3})\)')

    def __init__(self, db_name="matstract_db", local=True):
        db = "production" if db_name == "matstract_db" else "testing"
        self._db = AtlasConnection(local=local, db=db).db
        self.parser = parsing.MaterialParser()
        self.simple_parser = parsing.SimpleParser()
        self.mat_list = []
        self.elem_name_dict = dict()
        for i, elem in enumerate(self.ELEMENTS):
            self.elem_name_dict[self.ELEMENT_NAMES[i]] = elem

    """
    Provides tools for converting the data in the database to suitable
    format for machine learning tasks
    """
    def to_word2vec_corpus(self, filename="abstracts", limit=None, newlines=False, line_per_abstract=True, doi=None,
                           only_relevant=False, exclude_punct=False, year_max=None, split_years=False, convert_num=True,
                           normalize_mats=True):
        """

        :param filename:
        :param limit:
        :param newlines:
        :param line_per_abstract:
        :param doi:
        :param only_relevant:
        :param exclude_punct:
        :param year_max:
        :param split_years:
        :param normalize_mats:
        :return:
        """
        if limit:
            sample = True
        else:
            sample = False
        abstracts = self._get_abstracts(
            sample=sample,
            col=self.TOK_ABSTRACT_COL,
            doi=doi,
            year_max=year_max)

        if line_per_abstract:
            nl_tok = ""
        elif newlines:
            nl_tok = "\n"
        else:
            nl_tok = ""

        if not limit:
            total = abstracts.count()
        else:
            total = limit
        i = 0
        pbar = tqdm(total=total)
        for abstract in abstracts:
            if split_years and abstract["year"]:
                writefile = filename+"_"+str(abstract["year"])
            else:
                writefile = filename
            with open(writefile, "a+") as abstract_file:
                if i < total:
                    # processesed_abs = {"doi": abstract["doi"], "title": [], "abstract": []}
                    abs_txt = ""
                    ttl = abstract[self.TTL_FILED]
                    abs = abstract[self.ABS_FIELD]
                    if not only_relevant or self.is_relevant(abs):
                        if ttl:
                            for sentence in ttl:
                                abs_txt += " ".join(self.process_sentence(
                                    sentence, exclude_punct, convert_num=convert_num, normalize_mats=normalize_mats
                                )[0] + [nl_tok])
                                # processesed_abs["title"].append(self.process_sentence(sentence, exclude_punct)[0])
                        if abs:
                            for sentence in abs:
                                abs_txt += " ".join(self.process_sentence(
                                    sentence, exclude_punct, convert_num=convert_num, normalize_mats=normalize_mats
                                )[0] + [nl_tok])
                                # processesed_abs["abstract"].append(self.process_sentence(sentence, exclude_punct)[0])
                        if line_per_abstract:
                            abs_txt += "\n"
                        abstract_file.write(abs_txt)
                        # getattr(self._db, self.TOK_ABSTRACT_COL+"_processed").insert_one(processesed_abs)
                        i += 1
                        pbar.update(1)
                else:
                    break
        pbar.close()
        with open(filename+'_formula.pkl', 'wb') as f:
            pickle.dump(self.material_counts(), f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def tokenize(text):
        """
        Returns a list of tokens using chemdataextractor tokenizer.
        Keeps the structure of sentences.
        """
        cde_p = Paragraph(text)
        tokens = cde_p.tokens
        toks = []
        for sentence in tokens:
            toks.append([])
            for tok in sentence:
                toks[-1].append(tok.text)
        return toks

    def text2sent(self, text):
        """
        Same as tokenize, except doesn't respect the individual sentences in text
        :param text: a single string
        :return:
        """
        sentence = []
        for sent in self.tokenize(text):
            sentence += sent
        return sentence

    def tokenize_abstracts(self, limit=None, override=False, sample=False):

        # get the abstracts
        abstracts = self._get_abstracts(limit=limit, sample=sample)
        existing_dois = [abstr[self.DOI_FIELD] for abstr in self._get_abstracts(col=self.TOK_ABSTRACT_COL)]

        count = abstracts.count() if limit is None else limit

        def insert_abstract(a):
            if override or (not override and a[self.DOI_FIELD] not in existing_dois):
                # saving time by not tokenizing the text if abstract already exists
                try:
                    abs_tokens = {
                        self.DOI_FIELD: a[self.DOI_FIELD],
                        self.TTL_FILED: self.tokenize(a[self.TTL_FILED]),
                        self.ABS_FIELD: self.tokenize(a[self.ABS_FIELD]),
                        "abstract_id": a["_id"],
                    }
                except Exception as e:
                    print("Exception type: %s, doi: %s" % (type(e).__name__, a[self.DOI_FIELD]))
                    abs_tokens = {
                        self.DOI_FIELD: a[self.DOI_FIELD],
                        self.TTL_FILED: None,
                        self.ABS_FIELD: None,
                        "abstract_id": a["_id"],
                        "error": "%s: %s " % (type(e).__name__, str(e))
                    }
                if override:
                    # getattr(self._db, self.TOK_ABSTRACT_COL).insert_one(abs_tokens)
                    getattr(self._db, self.TOK_ABSTRACT_COL).replace_one({
                        "doi": a[self.DOI_FIELD]},
                        abs_tokens,
                        upsert=True)
                else:
                    try:
                        # we have already filtered so there should not be doi overlap
                        getattr(self._db, self.TOK_ABSTRACT_COL).insert_one(abs_tokens)
                    except Exception as e:
                        print("Exception type: %s, doi: %s" % (type(e).__name__, a[self.DOI_FIELD]))

        # tokenize and insert into the new collection (doi as unique key)
        for abstract in tqdm(abstracts, total=count):
            insert_abstract(abstract)
        #
        # with Pool() as p:
        #     list(tqdm(p.imap(insert_abstract, abstracts), total=count))

    def _get_abstracts(self, limit=None, col=None, doi=None, year_max=None, sample=True):
        """
        Returns a cursor of abstracts form mongodb
        :param limit:
        :return:
        """
        conditions = dict()
        if doi is not None:
            conditions["doi"] = {"$in": doi}
        if year_max is not None:
            conditions["year"] = {"$lt": year_max + 1}
        if col is None:
            col = self.RAW_ABSTRACT_COL

        if limit is not None:
            if sample:
                pipeline = [
                    {"$match": conditions},
                    {"$sample": {"size": limit}}
                ]
                abstracts = getattr(self._db, col).aggregate(pipeline, allowDiskUse=True)
            else:
                abstracts = getattr(self._db, col).find(conditions).limit(limit)
        else:
            if sample:
                size = getattr(self._db, col).find(conditions).count()
                pipeline = [
                    {"$match": conditions},
                    {"$sample": {"size": size}}
                ]
                abstracts = getattr(self._db, col).aggregate(pipeline, allowDiskUse=True)
            else:
                abstracts = getattr(self._db, col).find(conditions)
        return abstracts

    def is_number(self, t):
        return self.NR_BASIC.match(t.replace(',', '')) is not None

    # @staticmethod
    def process_sentence(self, s, exclude_punct=False, convert_num=True, normalize_mats=True):
        """
        processes sentence before word2vec training (no phrases)
        :param s: list of tokens
        :param exclude_punct: a bool to exlcude punctiation
        :param convert_num: a bool to convert numbers to <nUm>
        :param normalize_mats: a book to convert materials into their normalized form
        :return:
        """
        st = []
        split_indices = []
        for i, tok in enumerate(s):
            if exclude_punct and tok in self.PUNCT:
                continue
            elif convert_num and self.is_number(tok):
                try:
                    if s[i-1] == "(" and s[i+1] == ")" or s[i-1] == "〈" and s[i+1] == "〉":
                        pass
                    else:
                        tok = "<nUm>"
                except:
                    tok = "<nUm>"  # replace all numbers with a string <nUm>
            else:
                elem_with_valence = self.ELEMENT_VALENCE_IN_PAR.match(tok)
                if elem_with_valence is not None:
                    # change element name to symbol
                    elem_mention = elem_with_valence.group(1)
                    try:
                        formula = self.elem_name_dict[elem_mention.lower()]
                        matmention = elem_mention.lower()
                    except:
                        formula = elem_mention  # this was already the symbol
                        matmention = elem_mention
                    self.mat_list.append((matmention, formula))  # exclude the valence state from name
                    # split this for word2vec
                    st.append(matmention)
                    split_indices.append(i)
                    tok = elem_with_valence.group(2)
                elif tok in self.ELEMENTS_AND_NAMES:  # add element names to formulae
                    try:
                        formula = self.elem_name_dict[tok.lower()]
                        matmention = tok.lower()
                        tok = matmention
                    except:
                        formula = tok  # this was already the symbol
                        matmention = tok
                    self.mat_list.append((matmention, formula))
                elif self.is_simple_formula(tok):
                    formula = self.get_norm_formula(tok)
                    self.mat_list.append((tok, formula))
                    if normalize_mats:
                        tok = formula
                elif (len(tok) == 1 or (len(tok) > 1 and tok[0].isupper() and tok[1:].islower())) \
                        and tok not in self.ELEMENTS and tok not in self.UNITS \
                        and self.ELEMENT_DIRECTION_IN_PAR.match(tok) is None:
                    # to lowercase if only first letter is uppercase (chemical elements already covered above)
                    tok = deaccent(tok.lower()) if len(tok) > 1 else tok.lower()
                else:
                    # splitting units from numbers (e.g. you can get 2mol., 3V, etc..)
                    nr_unit = self.NR_AND_UNIT.match(tok)
                    if nr_unit is None or nr_unit.group(2) not in self.UNITS:
                        tok = deaccent(tok)  # matches the pattern but not in the list of units
                    else:
                        # splitting the unit from number
                        st.append("<nUm>") if convert_num else st.append(nr_unit.group(1))
                        split_indices.append(i)
                        tok = nr_unit.group(2)  # the unit
            st.append(tok)
        return st, split_indices

    def material_counts(self):
        counts = dict()
        for mat in self.mat_list:
            if mat[1] not in counts:
                counts[mat[1]] = dict()
                counts[mat[1]][mat[0]] = 1
            elif mat[0] not in counts[mat[1]]:
                counts[mat[1]][mat[0]] = 1
            else:
                counts[mat[1]][mat[0]] += 1
        return counts

    def is_simple_formula(self, text):
        if self.VALENCE_INFO.search(text) is not None:
            # 2 consecutive II, IV or VI should not be parsed as formula
            # related to valence state, so dont want to mix with I and V elements
            return False
        elif any(char.isdigit() or char.islower() for char in text):
            # has to contain at least one lowercase letter or at least one number (to ignore abbreviations)
            try:
                if text in ["O2", "N2", "Cl2", "F2", "H2"]:
                    # including chemical elements that are diatomic at room temperature and atm pressure,
                    # despite them having only a single element
                    return True
                composition = Composition(text)
                # has to contain more than one element
                if len(composition.keys()) < 2 or any([not self.simple_parser.is_element(key) for key in composition.keys()]):
                    return False
                # if the simple parser passes, try the more advanced one to catch more mistakes
                if any([not self.simple_parser.is_element(key) for key in self.parser.parse_formula(text).keys()]):
                    return False
                return True
            except Exception:
                return False
        else:
            return False

    def get_ordered_integer_formula(self, el_amt, max_denominator=1000):
        # return alphabetically ordered formula with integer fractions
        g = gcd_float(list(el_amt.values()), 1 / max_denominator)
        d = {k: round(v / g) for k, v in el_amt.items()}
        formula = ""
        for k in sorted(d):
            if d[k] > 1:
                formula += k + str(d[k])
            elif d[k] != 0:
                formula += k
        return formula

    def get_norm_formula(self, text, max_denominator=1000):
        try:
            # using Olga's parser
            formula_dict = dict(self.parser.parse_formula(text))
            for key in formula_dict:
                formula_dict[key] = float(formula_dict[key])
            integer_formula = self.get_ordered_integer_formula(formula_dict, max_denominator)
            return integer_formula
        except Exception:
            return text

