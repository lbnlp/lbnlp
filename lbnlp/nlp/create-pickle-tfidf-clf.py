import pymongo
import urllib
import re
from matscholar_core.nlp import relevance
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
import dill as pickle
from sklearn import model_selection

'# FILL IN YOUR CREDENTIALS'
date_format = re.compile('(1|2)[0-9]{3}')
client = pymongo.MongoClient('YOUR_DB_ADDRESS', username='YOUR_USERNAME', password='YOUR_PASSWORD', authSource='YOUR_SOURCE')
db = client['matscholar_prod']

'#LABELED - FILL IN YOUR CREDENTIALS'
client_ann = pymongo.MongoClient('YOUR_ANN_BD_LINK' % (urllib.parse.quote_plus("USERNAME"), urllib.parse.quote_plus("PASSWORD")), authSource='SOURCE')
db_ann = client_ann.matstract_db.macro_ann
dois_ann_full = [(e['doi'],e['relevant'],e['user']) for e in db_ann.find({'relevant': {'$in':[True, False]}})]

'#GET FULL LABELED DATABASE WITH ALL ABSTRACTS'
names_full = {'abstract': [],'relevance': [], 'doi': []}
abstracts_ann_full = pd.DataFrame(names_full)
for line in dois_ann_full:
    abs_ann = [e['abstract'] for e in db['entries'].find({'doi':line[0]})]
    if len(abs_ann) > 0:
        new_row = {'abstract': abs_ann[0], 'relevance': line[1], 'doi': line[0]}
        abstracts_ann_full = abstracts_ann_full.append(new_row, ignore_index=True)

#print(len(abstracts_ann_full[abstracts_ann_full['relevance']==True]))
#print(len(abstracts_ann_full[abstracts_ann_full['relevance']==False]))

data_train = abstracts_ann_full['abstract']
y_train = abstracts_ann_full['relevance']

#abstracts_ann_full.to_csv('abstracts_ann_full.csv', sep='\t')

'#PREPROCESSING & TOKENIZATION'
relClass = relevance.RelevanceClassifier()
processed_train = pd.Series([relClass._preprocess(doc) for doc in data_train])

'# TO TURN OFF SKLEARN TOKENIZER AND PREPROCESSOR AS IT IS DONE BY MATSCHOLARPROCESS'
def dummy_fun(doc):
    return doc

'#NLP - TF-IDF FROM SKLEARN'
vectorizer = TfidfVectorizer(analyzer='word', binary=False, decode_error='strict', encoding='utf-8', input='content', lowercase=True, max_df=1.0, max_features=None, min_df=1, ngram_range=(1, 1), norm='l2', preprocessor=dummy_fun, smooth_idf=True, stop_words='english', strip_accents=None, sublinear_tf=False, token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=dummy_fun, use_idf=True, vocabulary=None)
X_train = vectorizer.fit_transform(processed_train)

'#ML - BEST AMONG ALL TESTED IS RIDGE. HERE BEST PARAMETERS ARE USED'
clf = RidgeClassifier(tol=0.0125, solver="sag", alpha=1.1)
clf.fit(X_train, y_train)

'#SAVE TRAINED MODELS AS PICKLE TO LOAD IN MATSCHOLAR_CORE/NLP/RELEVANCE.PY'
with open('models/tfidf_new.p', 'wb') as file:
    pickle.dump(vectorizer, file)

with open('models/relevance_model_new.p', 'wb') as file:
    pickle.dump(clf, file)

print(model_selection.cross_val_score(clf, X_train,y_train,cv=7).mean())
