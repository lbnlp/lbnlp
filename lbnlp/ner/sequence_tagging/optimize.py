import pickle
import numpy as np
import os
import multiprocessing
import shutil

from train_test import run

def get_metrics(log_file = 'log.txt'):
    with open(log_file) as f:
        lines = [line.strip() for line in f]
    best_score_line = 0
    for n, line in enumerate(lines):
        if 'new best score!' in line:
            best_score_line = n
    best_f1 = float(lines[best_score_line-1].split('f1')[-1])
    best_acc = float(lines[best_score_line-1].split('acc')[-1].split()[0])
    return best_acc, best_f1

def worker(param_set):
    log_name = 'logs/{}_{}_{}_{}'.format(*param_set.values())

    run(param_set['learning_rate'],
        param_set['dropout'],
        param_set['word_lstm_size'],
        param_set['char_lstm_size'],
        log_file=log_name)

    acc, f1 = get_metrics(log_name)
    with open('out.txt', 'a') as f:
        print('accuracy: {}; f1: {}; word_size: {}; char_size: {}; dropout: {}; learning_rate: {}'.format(
            acc, f1, *param_set.values()
        ), file=f)

if __name__ == '__main__':
    shutil.rmtree(('logs'))
    os.mkdir('logs')

    filename_dev = "data/my_data/lstm_test.txt"
    filename_test =  "data/my_data/lstm_test.txt"
    filename_train =  "data/my_data/lstm_train.txt"

    annotations = pickle.load(open('annotations_np.p', 'rb'))

    # Get the train/test set
    cutoff = int(len(annotations)*0.85)
    annotations_train = annotations[:cutoff]
    annotations_test = annotations[cutoff:]

    # Print the train set
    with open(filename_train, 'w+') as f:
        for doc in annotations_train:
            for sent in doc:
                for (word, pos), bio in sent:
                    print('{} {}'.format(word, bio), file=f)
                print('\n', file=f)

    # Print the test set
    with open(filename_test, 'w+') as f:
        for doc in annotations_test:
            for sent in doc:
                for (word, pos), bio in sent:
                    print('{} {}'.format(word, bio), file=f)
                print('\n', file=f)


    word_lstm_size = [100, 200, 300]
    char_lstm_size = [30, 60]
    dropout = np.random.uniform(0.2, 0.7, size = 8)
    learning_rate = np.random.uniform(0.001, 0.03, size = 15)
    param_sets = [{'word_lstm_size': ws, 'char_lstm_size': cs, 'dropout': d, 'learning_rate': lr}
                  for ws in word_lstm_size
                  for cs in char_lstm_size
                  for d in dropout
                  for lr in learning_rate]


    with open('out.txt', 'w+') as f:
        print('beginning...', file =f)

    jobs = []
    for param_set in param_sets:
        p = multiprocessing.Process(target=worker, args=(param_set,))
        jobs.append(p)
        p.start()

        for job in jobs:
            job.join()











