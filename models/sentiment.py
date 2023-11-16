import logging
import pickle
import string

import pandas as pd
import spacy
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from common.tokenizer import tokenize

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()


# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

    # Removing stop words
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

    # return preprocessed list of tokens
    return mytokens


def main():
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s', level=logging.INFO)
    # connect()

    df = pd.read_csv('data/training.1600000.processed.noemoticon.csv')
    df['target'] = df['target'].apply(lambda x: int(x) >= 3)

    bow_vector = CountVectorizer(tokenizer=tokenize, ngram_range=(1, 1))
    tfidf_vector = TfidfVectorizer(tokenizer=tokenize)

    X = df['text']  # the features we want to analyze
    ylabels = df['target']  # the labels, or answers, we want to test against

    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

    classifier = LogisticRegression()

    # Create pipeline using Bag of Words
    pipe = Pipeline([('vectorizer', tfidf_vector),
                     ('classifier', classifier)])

    # model generation
    pipe.fit(X_train, y_train)

    # Predicting with a test dataset
    predicted = pipe.predict(X_test)

    # Model Accuracy
    print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted))
    print("Logistic Regression Precision:", metrics.precision_score(y_test, predicted))
    print("Logistic Regression Recall:", metrics.recall_score(y_test, predicted))

    pickle.dump(classifier, open('model.pickle', 'wb'))


if __name__ == '__main__':
    main()
