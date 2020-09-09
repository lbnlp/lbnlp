import unittest
import numpy.testing as npt
from matscholar_core.nlp.word_embeddings import EmbeddingEngine


class EmbeddingEngineTest(unittest.TestCase):

    def setUp(self):

        self.ee = EmbeddingEngine()

    def test_make_phrases(self):

        phrases = self.ee.make_phrases(["The", "thermoelectric", "figure",
                                        "of", "merit", "is", "unknown."], reps=1)
        self.assertListEqual(['The', 'thermoelectric_figure', 'of',
                              'merit', 'is', 'unknown.'], phrases)

        phrases2 = self.ee.make_phrases(["The", "thermoelectric", "figure",
                                        "of", "merit", "is", "unknown."], reps=2)
        self.assertListEqual(['The', 'thermoelectric_figure_of_merit',
                              'is', 'unknown.'], phrases2)

    def test_get_word_vector(self):

        word_vector, pwp = self.ee.get_embedding("thermoelectric")

        known_word_vector = [ 1.0962e-01,  2.3682e-02,  3.4637e-02,  9.2926e-03, -2.8290e-02,  2.6611e-02,
                              1.7078e-01, -1.2018e-01,  1.3458e-02,  2.0105e-01, -3.2806e-02,  1.4381e-02,
                             -2.4200e-02, -1.1055e-02,  4.6143e-02,  5.2124e-02, -2.4307e-02,  2.1942e-02,
                             -1.9012e-02,  7.5073e-02,  7.9956e-02, -5.7007e-02, -1.6663e-02, -3.7689e-02,
                              1.2028e-04, -6.3171e-02,  3.7689e-02,  1.6663e-01,  1.1670e-01,  1.5295e-01,
                             -2.9617e-02, -3.3325e-02,  1.3626e-02,  8.4045e-02, -1.5671e-02, -1.8616e-01,
                             -1.0687e-01, -3.6743e-02, -1.0278e-01,  2.9785e-02,  1.3196e-01, -4.1321e-02,
                             -1.7407e-01, -9.7275e-03, -4.5105e-02, -2.7939e-02, -6.2408e-02, -3.9093e-02,
                             -1.0266e-01, -1.7776e-02,  5.1605e-02,  7.1533e-02, -2.4700e-03, -2.9388e-02,
                             -6.6345e-02,  5.7068e-02,  9.8694e-02, -4.7852e-02, -3.3234e-02, -8.5632e-02,
                              5.2490e-02,  1.0303e-01, -4.0009e-02,  3.9215e-02,  1.3247e-03, -4.5654e-02,
                              2.5120e-03,  8.6792e-02, -4.2755e-02,  1.5234e-01,  3.6438e-02,  7.3776e-03,
                             -9.9182e-02, -8.7433e-03,  2.8107e-02,  2.6199e-02, -4.7943e-02, -8.8623e-02,
                             -8.6731e-02, -4.9858e-03,  3.5461e-02,  6.8726e-02,  6.8054e-02, -5.7007e-02,
                              7.9163e-02, -3.9721e-04,  6.3843e-02,  7.7759e-02,  5.2216e-02, -2.0554e-02,
                              2.9541e-02,  1.7139e-01,  7.2083e-02,  7.3730e-02, -8.0200e-02,  1.0078e-02,
                              1.3037e-01,  4.2847e-02, -1.8585e-02,  1.4929e-01,  1.4209e-01, -3.4363e-02,
                             -4.6783e-02,  1.0651e-02, -1.1542e-01, -2.7969e-02, -2.5650e-02,  4.7073e-03,
                              1.1505e-01,  3.8574e-02, -5.5450e-02, -6.1646e-03, -1.0754e-01,  3.4454e-02,
                             -3.9642e-02,  8.6243e-02, -1.1829e-01, -9.3994e-02, -7.3975e-02,  1.6876e-02,
                              2.5696e-02, -4.0283e-02,  9.7168e-02, -1.6321e-01, -4.3249e-04,  2.0569e-02,
                              8.7463e-02, -9.1187e-02,  3.5675e-02, -6.1066e-02, -5.2155e-02, -4.0009e-02,
                              3.9482e-03,  5.9784e-02,  3.2104e-02,  4.0009e-02,  9.2468e-02,  4.9744e-02,
                              9.7322e-04,  8.2474e-03, -7.4280e-02, -2.7969e-02, -2.3331e-02,  1.0095e-01,
                             -9.4666e-02,  1.0986e-02,  2.4277e-02, -3.6652e-02,  6.7078e-02,  8.1238e-02,
                              8.0994e-02, -8.1299e-02,  6.2500e-02, -9.4177e-02, -9.8145e-02,  7.8918e-02,
                             -7.3669e-02, -4.4594e-03,  1.2039e-02,  1.2329e-02, -2.0813e-02,  7.0858e-04,
                              4.7607e-02, -6.2622e-02,  4.3793e-02,  2.4643e-03, -4.9561e-02,  6.6650e-02,
                             -6.6032e-03,  6.1920e-02, -3.9459e-02, -7.6904e-02,  5.7106e-03,  6.6589e-02,
                             -1.0040e-01, -2.2049e-02,  9.6802e-02, -3.0609e-02,  3.3875e-02, -3.7048e-02,
                             -5.5145e-02,  1.0260e-01, -5.4626e-02, -2.1454e-02,  7.0068e-02,  2.3697e-02,
                             -4.7302e-02,  7.0801e-02,  3.1219e-02, -1.0034e-01, -1.6586e-02, -1.8509e-02,
                              9.1797e-02, -5.2002e-02, -4.1351e-02, -7.5439e-02, -8.3374e-02, -8.3130e-02,
                             -1.5100e-01, -3.5248e-02 ]

        self.assertListEqual([["thermoelectric"]], pwp)
        npt.assert_almost_equal(word_vector, [known_word_vector], decimal=4)

        word_vector, pwp = self.ee.get_embedding(
            ["thermoelectric asdfsdfdsfsdf", "thermoelectric"],
            ignore_missing=True)
        self.assertListEqual([["thermoelectric"], ["thermoelectric"]], pwp)
        npt.assert_almost_equal(word_vector, [known_word_vector, known_word_vector], decimal=4)

    def test_close_words(self):

        # example 1
        close_words, close_word_scores, pp, pn = self.ee.close_words("Thermoelectric", top_k=8)

        known_close_words = ['thermoelectrics', 'thermoelectric_properties',
                             'thermoelectric_power_generation',
                             'thermoelectric_figure_of_merit', 'seebeck_coefficient',
                             'thermoelectric_generators', 'figure_of_merit_ZT',
                             'thermoelectricity']
        known_close_word_scores = [0.8433, 0.834, 0.7935, 0.7915,0.7754, 0.764, 0.759, 0.7515]

        self.assertListEqual(close_words, known_close_words)
        self.assertListEqual(pp, ["thermoelectric"])
        self.assertListEqual(pn, [])
        npt.assert_almost_equal(close_word_scores, known_close_word_scores,
                                decimal=3)

        # example 2
        close_words, close_word_scores, pp, pn = self.ee.close_words(positive=["Fe", "cobalt"],
                                                                     negative=["iron"],
                                                                     top_k=8)

        known_close_words = ['Co', 'Ni', 'Mn', 'Cr', 'Cu', 'Ru', 'Zn', 'ferrites_MFe2O4']
        known_close_word_scores = [0.937, 0.832, 0.75, 0.717, 0.71, 0.67, 0.662, 0.66]

        self.assertListEqual(close_words, known_close_words)
        self.assertListEqual(pp, ["Fe", "cobalt"])
        self.assertListEqual(pn, ["iron"])
        npt.assert_almost_equal(close_word_scores, known_close_word_scores,
                                decimal=3)

    def test_find_similar_materials(self):

        sentence = "A material with a high thermoelectric figure of merit is"

        similar, processed_positive, processed_negative = \
            self.ee.find_similar_materials(sentence, min_count=10, use_output_emb=False, ignore_missing=True)

        self.assertEqual(len(similar), 11319)
        self.assertEqual(processed_positive, ["A material with a high thermoelectric_figure_of_merit is".split()])

        similar = similar[0:10]
        similar_mats = [s[0] for s in similar]
        similar_scores = [s[1] for s in similar]

        known_similar = [('In', 0.671),
                         ('K', 0.623),
                         ('BiCePt', 0.587),
                         ('La6Mn9O30Sr4V', 0.57),
                         ('La7Mn8O24Sr', 0.569),
                         ('Ni2PrSi2', 0.568),
                         ('At', 0.561),
                         ('Ni4SiSm', 0.561),
                         ('As', 0.56),
                         ('CeOPRu', 0.559)]
        known_similar_mats = [s[0] for s in known_similar]
        known_similar_scores = [s[1] for s in known_similar]

        self.assertListEqual(similar_mats, known_similar_mats)
        npt.assert_almost_equal(similar_scores, known_similar_scores,
                                decimal=3)

        similar_out, _, __ = \
            self.ee.find_similar_materials(sentence, min_count=10, use_output_emb=True, ignore_missing=True)

        similar_out = similar_out[0:10]
        similar_mats_out = [s[0] for s in similar_out]
        similar_scores_out = [s[1] for s in similar_out]

        known_similar_out = [('Tc', -0.121),
                             ('BeLi', -0.142),
                             ('BiCuOSe', -0.15),
                             ('BFeO3', -0.15),
                             ('BO3V', -0.151),
                             ('InSe2Tl', -0.152),
                             ('AgSbTe2', -0.157),
                             ('CuO2', -0.159),
                             ('SSm', -0.159),
                             ('AlFe2V', -0.162)]

        known_similar_mats_out = [s[0] for s in known_similar_out]
        known_similar_scores_out = [s[1] for s in known_similar_out]

        self.assertListEqual(similar_mats_out, known_similar_mats_out)
        npt.assert_almost_equal(similar_scores_out, known_similar_scores_out,
                                decimal=3)

    def test_most_common_form(self):

        thermo_mats, _, __ = self.ee.find_similar_materials(["thermoelectric"], use_output_emb=True, min_count=3)
        common_forms = self.ee.most_common_form(thermo_mats[:3])

        self.assertEqual(["Bi2Te3", "MgAgSb", "PbTe"], [cf[0] for cf in common_forms])
        npt.assert_array_almost_equal([0.19262797, 0.13790667, 0.12772012], [cf[1] for cf in common_forms])
        self.assertEqual([2452, 9, 2598], [cf[2] for cf in common_forms])

    def test_filter_by_elements(self):

        formulas = [('Tc', -0.10358655),
                    ('BiCuOSe', -0.1120525),
                    ('AgSbTe2', -0.11389587),
                    ('BeLi', -0.11680989),
                    ('AgSbSe2', -0.11982395),
                    ('PbTe', -0.1204319),
                    ('AlFe2V', -0.12064656),
                    ('InSe2Tl', -0.1227376),
                    ('BFeO3', -0.12427175),
                    ('PbSnTe', -0.12894371)]

        self.assertEqual(
            self.ee.filter_by_elements(formulas, plus_elems=["Te"]),
            [('AgSbTe2', -0.11389587),
             ('PbTe', -0.1204319),
             ('PbSnTe', -0.12894371)]
        )

        self.assertEqual(
            self.ee.filter_by_elements(formulas, plus_elems=["Te", "Bi"]),
            [('BiCuOSe', -0.1120525),
             ('AgSbTe2', -0.11389587),
             ('PbTe', -0.1204319),
             ('PbSnTe', -0.12894371)]
        )

        self.assertEqual(
            self.ee.filter_by_elements(formulas, minus_elems=["O", "Se"]),
            [('Tc', -0.10358655),
             ('AgSbTe2', -0.11389587),
             ('BeLi', -0.11680989),
             ('PbTe', -0.1204319),
             ('AlFe2V', -0.12064656),
             ('PbSnTe', -0.12894371)]
        )

        self.assertEqual(
            self.ee.filter_by_elements(formulas, plus_elems=["Pb"], minus_elems=["O", "Se"]),
            [('PbTe', -0.1204319),
             ('PbSnTe', -0.12894371)]
        )

        self.assertEqual(
            self.ee.filter_by_elements(formulas, plus_elems=["O"], minus_elems=["O", "Se"]),
            [('Tc', -0.10358655),
             ('AgSbTe2', -0.11389587),
             ('BeLi', -0.11680989),
             ('PbTe', -0.1204319),
             ('AlFe2V', -0.12064656),
             ('BFeO3', -0.12427175),
             ('PbSnTe', -0.12894371)]
        )

    def test_mentioned_with(self):
        self.assertTrue(self.ee.mentioned_with("Bi2Te3", ["thermoelectric", "ZT", "power_factor", "asdasdasdvsdf"]))
        self.assertFalse(self.ee.mentioned_with("Cu7Te5", ["thermoelectric", "asdasdasd"]))
