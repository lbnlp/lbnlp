from model.data_utils import CoNLLDataset
import os
import build_data
from model.ner_model import NERModel
from model.config import Config

class NERTrainer(object):

    def run(self, learning_rate=0.012, dropout=0.45, word_size=200, char_size=50):
        build_data.main('log.txt')
        config, model = self._train(learning_rate, dropout, word_size, char_size)
        metrics = self._evaluate(config, model)
        model.save_final_session()
        model.close_session()
        return metrics

    def _train(self, lr, dropout, word_size, char_size):
        # create instance of config
        config = Config()

        # Hyperparameters
        config.lr = lr
        config.dropout = dropout
        config.dim_word = word_size
        config.dim_char = char_size

        # build model
        model = NERModel(config)
        model.build()

        # create datasets
        dev = CoNLLDataset(config.filename_dev, config.processing_word,
                           config.processing_tag, config.max_iter)
        train = CoNLLDataset(config.filename_train, config.processing_word,
                             config.processing_tag, config.max_iter)

        # train model
        model.train(train, dev)

        return config, model

    def _evaluate(self, config, model, test_set='dev'):

        # create dataset
        if test_set == 'dev':
            _test_set = config.filename_dev
        else:
            _test_set = config.filename_test
        test = CoNLLDataset(_test_set, config.processing_word,
                            config.processing_tag, config.max_iter)

        # evaluate and interact
        metrics = model.evaluate(test)
        f1 = metrics['f1']
        acc = metrics['acc']
        all_entities = model.evaluate_final(test)
        return f1, acc, all_entities

class NERClassifier(object):

    MODEL_DIR = os.path.join(os.path.dirname(__file__), '../results/crf/final_model.weights/')

    def __init__(self):
        self.config = Config()
        self.config.dim_word = 250
        self.config.dim_char = 50
        self.model = NERModel(self.config)
        self.model.build()
        self.model.restore_session(self.MODEL_DIR)

    def _evaluate(self, test_set_loc):
        test = CoNLLDataset(test_set_loc, self.config.processing_word,
                            self.config.processing_tag, self.config.max_iter)
        # evaluate and interact
        metrics = self.model.evaluate(test)
        f1 = metrics['f1']
        acc = metrics['acc']
        all_entities = self.model.evaluate_final(test)
        return f1, acc, all_entities

    def assess(self, set='test'):
        """
        Calculates the model accuracy metrics for the train, dev, and test sets
        """

        # Evaluate the training set
        train_loc = self.config.filename_train
        train_metrics = self._evaluate(train_loc)

        # Evaluate on the dev set
        dev_loc = self.config.filename_dev
        dev_metrics = self._evaluate(dev_loc)

        # Evaluate on test set
        test_loc = self.config.filename_test
        test_metrics = self._evaluate(test_loc)

        return train_metrics, dev_metrics, test_metrics

if __name__ == '__main__':
    trainer = NERTrainer()
    trainer.run()




