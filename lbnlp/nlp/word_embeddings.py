import numpy as np
import operator
import regex
import warnings

from json import load
from gensim.models.phrases import Phraser
from pymagnitude import Magnitude
from os import path
import logging

from matscholar_core.processing.word2vec import DataPreparation
from matscholar_core.database import ElasticConnection

model_dir = path.join(path.dirname(__file__), "models/embeddings")

EMBEDDINGS = path.join(model_dir, 'embeddings.magnitude')
OUT_EMBEDDINGS = path.join(model_dir, 'out_embeddings.magnitude')
FORMULAS = path.join(model_dir, 'relevant_3273k_formula.json')
PHRASER = path.join(model_dir, 'phraser.pkl')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p')


class EmbeddingEngine:
    """
    An interface to query pre-trained word vectors.
    """

    ABBR_LIST = ["C41H11O11", "PV", "OPV", "PV12", "CsOS", "CsKPSV", "CsPS",
                 "CsHIOS", "OPV", "CsPSV", "CsOPV", "CsIOS", "BCsIS", "CsPrS",
                 "CEsH", "KP307", "AsOV", "CEsS", "COsV", "CNoO", "BEsF",
                 "I2P3", "KP115", "BCsIS", "C9705IS", "ISC0501", "B349S",
                 "CISe", "CISSe", "CsIPS", "CEsP", "BCsF", "CsFOS", "BCY10",
                 "C12P", "EsHP", "CsHP", "C2K8", "CsOP", "EsHS", "CsHS", "C3P",
                 "C50I", "CEs", "CSm", "BF", "EsN", "BN50S", "AsCP", "CPo",
                 "LiPb17", "CsS", "EsIS", "AsCU", "CCsHS", "CsHPU", "AsOS",
                 "AsCI", "EsF", "FV448", "CNS", "CP5", "AsFP", "EsOP", "NS",
                 "NS2", "EsI", "BH", "PPmV", "PSe", "AsN", "OPV5", "NSiW", "CsHHS"]

    def __init__(self,
                 embeddings_source=EMBEDDINGS,
                 out_embeddings_source=OUT_EMBEDDINGS,
                 formulas_source=FORMULAS,
                 phraser_source=PHRASER):
        """

        :param embeddings_source: can be instance of a Magnitude object
        or url or path to a serialized Magnitude object
        :param out_embeddings_source: can be instance of a Magnitude object
        or url or path to a serialized Magnitude object
        :param formulas_source: can be url or path to a JSON-serialized dict
        of formulae, if not supplied a default file is loaded
        """

        # hidden layer embeddings (W)
        self.embeddings = Magnitude(embeddings_source, eager=False)

        # output layer embeddings (O)
        self.out_embeddings = Magnitude(out_embeddings_source)

        # load pre-trained formulas from embeddings
        with open(formulas_source, 'r') as f:
            self.formulas_with_abbreviations = load(f)

        self.dp = DataPreparation(local=False)

        self.es = ElasticConnection()

        self.formulas = {k: v for k, v in self.formulas_with_abbreviations.items()
                         if k not in self.ABBR_LIST}

        self.formula_counts = {root_formula: sum(formulas.values())
                               for root_formula, formulas in self.formulas.items()}

        self.most_common_forms = {
            formula_group_name:
                (formula_group_name if formula_group_name in self.dp.ELEMENTS
                 else max(formulae.items(),
                          key=operator.itemgetter(1))[0])
            for formula_group_name, formulae in
            self.formulas_with_abbreviations.items()
        }

        self.phraser = Phraser.load(phraser_source)

    def make_phrases(self, sentence, reps=2):
        """
        generates phrases from a sentence of words
        :param sentence: a list of tokens
        :param reps: how many times to combine the words
        :return:
        """
        while reps > 0:
            sentence = self.phraser[sentence]
            reps -= 1
        return sentence

    def prepare_wordphrase(self, wp, im=False):
        """
        Process a string into words and phrases according to existing embeddings
        :param wp: the string to process
        :param im: if True, will ignore missing words, otherwise will generate random vectors
        :return: a list of processed words and phrases
        """
        processed_wp = self.make_phrases(self.dp.process_sentence(self.dp.text2sent(wp))[0])
        if im:
            processed_wp = [pwp for pwp in processed_wp if pwp in self.embeddings]
        return processed_wp

    def get_embedding(self, wordphrases, ignore_missing=False, normalized=True):
        """
        Gets the embedding for the given word
        :param wordphrases: a string or a list of strings to request embedding for
        :param ignore_missing: if true, will ignore missing words, otherwise will query them
        using pymagnitude defult out of dictionary handling
        :param normalized: if False, returns non-normalized embeddings (True by default)
        :return: an embedding matrix with each row corresponding to a single processed word or phrase
        taken from wordphrases, as well as the lists of processed wordphrases
        """
        def get_single_embedding(wp, im=ignore_missing, norm=normalized):
            """
            Returns a single embedding vector for the given string
            :param wp: a string to get a single embedding for
            :param im: boolen to ignore missing words or return some random vectors if False
            :param norm: if False, returns the non-normalized embedding (True by default)
            :return: a single embedding vector for the string (could be a composite embedding)
            """
            processed_wordphrase = self.prepare_wordphrase(wp, im)

            if len(processed_wordphrase) > 0:
                emb = np.mean(self.embeddings.query(
                    processed_wordphrase,
                    normalized=norm
                ), axis=0)
                if norm:
                    emb = emb / np.linalg.norm(emb)
                emb = emb.tolist()
            else:
                emb = [0] * self.embeddings.dim
            return emb, processed_wordphrase

        if not isinstance(wordphrases, list):
            wordphrases = [wordphrases]

        processed_wps = []
        embeddings = []

        try:
            for wordphrase in wordphrases:
                embedding, processed_wp = get_single_embedding(wordphrase, im=ignore_missing)
                processed_wps.append(processed_wp)
                embeddings.append(embedding)
        except Exception as ex:
            warnings.warn(ex)

        return embeddings, processed_wps

    def close_words(self, positive, negative=None, top_k=8, exclude_self=True, ignore_missing=True):
        """
        Returns a list of close words
        :param positive: can be either a string or a list of strings
        :param negative: same as word, but will be treated with a minus sign
        :param top_k: number of close words to return
        :param exclude_self: boolean, if the supplied word should be excluded or not
        :param ignore_missing: ignore words that are missing from the vocabulary
        :return: (words, scores, processed_positive, processed_negative)
        """

        if negative is None:
            negative = []
        else:
            if not isinstance(negative, list):
                negative = [negative]
        processed_negative = []
        for n in negative:
            processed_negative += self.prepare_wordphrase(n, im=ignore_missing)

        if not isinstance(positive, list):
            positive = [positive]
        processed_positive = []
        for p in positive:
            processed_positive += self.prepare_wordphrase(p, im=ignore_missing)

        most_similar = self.embeddings.most_similar(
            processed_positive,
            negative=processed_negative,
            topn=top_k)

        if not exclude_self:
            most_similar = [(processed_positive, 1.0)] + most_similar[:top_k-1]
        words, scores = map(list, zip(*most_similar))
        return words, [float(s) for s in scores], processed_positive, processed_negative

    def find_similar_materials(self, sentence, n_sentence=None, min_count=3,
                               use_output_emb=True, ignore_missing=True):
        """
        Finds materials that match the best with the context of the sentence
        :param sentence: a list of words
        :param n_sentence: a list of words for a negative context
        :param min_count: the minimum number of occurrences for the formula
        to be included
        :param use_output_emb: if True, use output layer embedding (O) instead of
        inner layer embedding (W)
        :return:
        """
        positive_embeddings, processed_sentence = \
            self.get_embedding(sentence, ignore_missing=ignore_missing)

        n_sentence = n_sentence or []
        negative_embeddings, processed_n_sentence = \
            self.get_embedding(n_sentence, ignore_missing=ignore_missing)

        emb = self.out_embeddings if use_output_emb else self.embeddings

        sum_embedding = np.sum(np.asarray(positive_embeddings), axis=0) - \
                        np.sum(np.asarray(negative_embeddings), axis=0)
        sum_embedding = sum_embedding / np.linalg.norm(sum_embedding)

        # formulas common enough to be above cut-off and that exist in embedding
        formulas = [f for f, count in self.formula_counts.items()
                    if (count > min_count) and (f in self.embeddings)]

        similarity_scores = np.dot(emb.query(formulas, normalized=True), sum_embedding)
        similarities = {f: float(similarity_scores[i]) for i, f in enumerate(formulas)}

        return sorted(similarities.items(), key=lambda x: x[1], reverse=True), processed_sentence, processed_n_sentence

    def most_common_form(self, formulas):
        """
        Return the most common form of the formula given a list with tuples
        [("normalized formula": score), ...]
        :param formulas: the dictionary
        :return: a list of common forms with counts, [("common form", score, counts in text), ...]
        """
        common_form_score_count = []
        for formula in formulas:
            if formula[0] in self.dp.ELEMENTS:
                most_common_form = formula[0]
            else:
                most_common_form = max(self.formulas[formula[0]].items(),
                                       key=operator.itemgetter(1))[0]
            common_form_score_count.append(
                (most_common_form, formula[1], sum(self.formulas[formula[0]].values()))
            )
        return common_form_score_count

    def filter_by_elements(self, formulas, plus_elems=None,
                           minus_elems=None, max=50):
        """
        Filter formulas according to the following rule: It has to have one of the plus_elements (if None all work),
        but it cannot have any of the minus_elems. If there is an overlap, the element is ignored
        :param formulas: a list of (formula, score) tuples
        :param plus_elems: the formula has to have at least one of these
        :param minus_elems: but cannot have any of these
        :param max: maximum number to return
        :return:
        """
        plus_elems = plus_elems or []
        minus_elems = minus_elems or []
        plus_elems, minus_elems = set(plus_elems) - set(minus_elems), set(minus_elems) - set(plus_elems)

        def has_plus(comp, pe):
            if pe is None or len(pe) == 0:
                return True
            for elem in comp:
                if elem in pe:
                    return True
            return False

        def has_minus(comp, me):
            if me is None or len(me) == 0:
                return False
            for elem in comp:
                if elem in me:
                    return True
            return False

        matched = 0
        matched_formula = []
        for form in formulas:
            composition = self.dp.parser.parse_formula(form[0])
            if has_plus(composition, plus_elems) and not has_minus(composition,
                                                                   minus_elems):
                matched_formula.append(form)
                matched += 1
            if matched >= max:
                return matched_formula
        return matched_formula

    def mentioned_with(self, material, words):
        """
        Returns True if the supplied material was mentioned with any of the words in any of the abstracts. This is a
        very strict text search and is aimed at high recall. This method is used for discovery so having higher recall
        might hinder some discoveries but will avoid too many false positives. E.g. for material=CuTe and
        words=["thermoelectric"], "CuTe2 is thermoelectric" will return True since "CuTe" will be matched with "CuTe2"
        in text search. The word search is exact, so if the keyword was "thermo" it would not match "thermoelectric".
        :param material: A material formula (does not have to be normalized)
        :param words: List of processed words and phrases (words separated by "_") to search the text for co-occurrences
        :return: True if the material is mentioned with any of the words, False otherwise
        """
        norm_material = self.dp.get_norm_formula(material) if self.dp.is_simple_formula(material) else material

        # different ways the material is written
        variations = self.formulas[norm_material] if norm_material in self.formulas else [norm_material]
        variations = "(" + " OR ".join(variations) + ")"
        targets = "(" + " OR ".join(words) + ")"
        query = "{} AND {}".format(targets, variations)
        if self.es.count_matches(query) > 0:
            return True
        else:
            return False


class Embedding(object):
    """
    A Mat2Vec word/phrase embedding. Embeddings are 200-dimensional vectors that represent the meaning
    of words and phrases in a mathematical way. Matscholar embeddings are derived from the Word2Vec
    NLP architecture (Mikolov et al., 2013). If no embedding exists for the single wordphrase, the sum of the
    embeddings of the sub-wordphrases is used.
    """

    def __init__(self, wordphrase, tag, embedding, compound):
        self.wordphrase = wordphrase
        self.tag = tag
        # ndarrays aren't json serializable, for now we'll just cast to a list.
        self.embedding = list(embedding)
        self.compound = compound


def number_to_substring(text, latex=False):
    return regex.sub("(\d*\.?\d+)", r'_\1', text) if latex else regex.sub("(\d*\.?\d+)", r'<sub>\1</sub>', text)
