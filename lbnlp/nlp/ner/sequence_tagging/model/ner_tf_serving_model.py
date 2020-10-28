import numpy as np
np.random.seed(1)
import os
import requests
import tensorflow as tf
tf.set_random_seed(1)

from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from .ner_model import NERModel

class NERServingModel(NERModel):
    """A variant of ner_model suitable for constructing and using a tf-serving API"""

    def __init__(self, config, api_url):
        super(NERServingModel, self).__init__(config)
        self.api_url = api_url

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            "word_ids": word_ids,
            "sequence_lengths": sequence_lengths
        }

        if self.config.use_chars:
            feed["char_ids"] = char_ids
            feed["word_lengths"] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed["labels"] = labels

        if lr is not None:
            feed["lr"] = lr

        if dropout is not None:
            feed["dropout"] = dropout

        return feed, sequence_lengths


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self._api_call_predict(fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            raise Exception

    def _api_call_predict(self,feed_dict):
        """ Make a call to a tf-serving server implementing the NER model

        Args:
            feed_dict: dictionary of inputs

        Returns:
            logits, trans_params
        """

        input_dict = {"inputs":feed_dict}
        r = requests.post(url=self.api_url, json=input_dict)
        if r.status_code == 200:
            r = r.json()
            return np.array(r['outputs']['logits']),np.array(r['outputs']['trans_params'])
        else:
            r.raise_for_status()

    def save_prediction_model(self,save_dir):
        """Makes a tf saved_model copy of the current NER model

        Args:
            save_dir: Directory to save the model

        Returns:
            self

        """

        tf.saved_model.simple_save(
        self.sess,
        save_dir,
        {"word_ids": self.word_ids,"sequence_lengths": self.sequence_lengths,"dropout":self.dropout,"word_lengths":self.word_lengths,"char_ids":self.char_ids},
        {"logits": self.logits,"trans_params":self.trans_params}
        )
        return self