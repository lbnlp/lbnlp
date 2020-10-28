import os

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word


class Config():

    def __init__(self, log_path='log.txt', load=True, train=False):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if train:
            if not os.path.exists(self.dir_output):
                os.makedirs(self.dir_output)

            if os.path.exists(log_path):
                os.remove(log_path)

        # create instance of logger
        self.logger = get_logger(log_path)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                                                   self.vocab_chars,
                                                   lowercase=True,
                                                   chars=self.use_chars)  # Set back to true
        self.processing_tag = get_processing_word(self.vocab_tags,
                                                  lowercase=False,
                                                  allow_unk=False)  # keep allow_unk = False

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed)
                           if self.use_pretrained else None)

    # general config
    dir_output = "results/crf/"
    dir_model = dir_output + "model.weights/"
    dir_final_model = dir_output + "final_model.weights/"
    path_log = dir_output + "log.txt"

    # embeddings
    dim_word = 200  # was 300, changed to 200
    dim_char = 20

    # glove files
    filename_glove = 'data/my_data/w2v_final.txt'  # 'data/my_data/w2v_3mil.txt' #data/my_data/w2v.txt'
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.6B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_dev = 'data/my_data/prop_dev.txt'  # lstm_dev_final.txt #'data/my_data/polymer_data/lstm_test.txt'
    filename_test = 'data/my_data/prop_test.txt'  # lstm_test_final.txt #"data/my_data/lstm_test.txt"  #test
    filename_train = 'data/my_data/prop_train.txt'  # lstm_train_final.txt'  #'data/my_data/polymer_data/lstm_train.txt'#"data/my_data/lstm_train_np.txt"#

    # filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs = 30
    dropout = 0.6  # 0.5
    batch_size = 128
    lr_method = "adam"
    lr = 0.012  # 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 4

    # model hyperparameters
    hidden_size_char = 50  # lstm on chars
    hidden_size_lstm = 250  # lstm on word embeddings #changed to 200, was 300

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    use_chars = True  # if char embedding, training is 3.5x slower on CPU



class Configure(Config):
    """
    Configuration class for NER model.
    """

    LOCAL_DIR = os.path.dirname(__file__)

    def __init__(self, data_dir=None):
        """
        Constructor method for Configure.
        """

        self.dir_data = self.LOCAL_DIR if not data_dir else data_dir

        # Model saving/loading
        self.dir_final_model = os.path.join(self.dir_data, "model.weights/")

        # vocabulary
        self.filename_words = os.path.join(self.dir_data, "words.txt")
        self.filename_tags = os.path.join(self.dir_data, "tags.txt")
        self.filename_chars = os.path.join(self.dir_data, "chars.txt")

        # Embeddings
        self.filename_glove = os.path.join(self.dir_data, "w2v.txt")
        self.filename_trimmed = os.path.join(self.dir_data, "glove.6B.{}d.trimmed.npz".format(200))

        # Output
        log_path = os.path.join(self.dir_data, "logs.txt")
        self.dir_output = os.path.join(self.dir_data, "results/")

        # Initialize parent class
        super().__init__(log_path=log_path)
