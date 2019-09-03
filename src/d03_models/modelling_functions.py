from collections import Counter
from datetime import datetime
import json
import jsonpickle
import os
import pickle
import re
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import nltk
from nltk import FreqDist, word_tokenize
# from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd
import preprocessor as p
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# functions for generating datafames to be used in modelling

def validate_train_test_split(df, test_size, validation_size, random_state):
    """
    Split dataframes into a train, test and validation datasets.
    Achieves this by first splitting the dataframe into a training and testing dataset,
    and then carves out a subset of the training data to use for validation.
    
    Parameters
    ----------
        df : dataframe to split 
        test_size: percent of data to use in the training dataset
        validation_size : the percent of training data to hold out as a validation set
        random_state : random seed to generate the two splits
        
    Returns
    -------
        train, test, validation (dataframes)
    """
    training_data, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, validation = train_test_split(training_data, test_size=validation_size, random_state=random_state)
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)
    print('Validation shape:', validation.shape)
    return train, test, validation

def generate_unbalanced_X_and_y(df, pos_class_size, random_state):
    """
    Generate a feature matrix X and target vector y with artificially 
    unbalanced classes through downsampling, where the minority class 
    is equal to a user-specified share of the majority class.
    
    Parameters
    ----------
        df : the dataframe with balanced classes (to be split into X and y); target feature
             must be labeled as "target"
        pos_class_size : the percentage of the positive class to retain, as a share of 
             the negative class (fraction between 0 and 1)
        random_state : random seed to generate the downsampling of the positive class
        
    Returns
    -------
        X, y
    """
    print('Original shape:', df.shape)

    # generate X
    X_neg = df[df.target == 0].drop(['target'], axis=True)
    print('X negative rows:', X_neg.shape)
    X_pos = df[df.target == 1].drop(['target'], axis=True).sample(n=round(len(X_neg)*pos_class_size), random_state=random_state)
    print(f'X positive rows {pos_class_size*100}% sample:', X_pos.shape)
    X = pd.concat([X_neg, X_pos], axis=0, sort=False)
    print('X new:', X.shape)
    
    # generate y
    y_neg = df[df.target == 0]['target']
    print('y negative rows:', y_neg.shape)
    y_pos = df[df.target == 1]['target'].sample(n=round(len(y_neg)*pos_class_size), random_state=random_state)
    print(f'y positive rows {pos_class_size*100}% sample:', y_pos.shape)
    y = pd.concat([y_neg, y_pos], axis=0, sort=False)
    print('y new:', y.shape)
    
    return X, y

def generate_word_vector_for_X(X_train, X_test, X_valid, tweet_colname, max_features):
    """
    Generate vectors of words from the text content of tweets, from three separate input 
    vectors for X (X_train, X_test and X_valid).
    
    Parameters
    ----------
        X_train : a feature matrix X for training that includes a column containing the text of tweets
        X_test : a feature matrix X for testing that includes a column containing the text of tweets
        X_valid : a feature matrix X for validation that includes a column containing the text of tweets
        tweet_colname : the name of the column that contains the text of tweets (string)
        max_features : the number of words to include in the vocabulary, only considering
            the top words ordered by term frequency across the corpus
    
    Returns
    -------
        X_train_content_feat, X_test_content_feat, X_valid_content_feat

    """
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = 'english',
                                 max_features = max_features,
                                 ngram_range = (1, 3))
    X_train_content = X_train[tweet_colname].tolist()
    X_test_content = X_test[tweet_colname].tolist()
    X_valid_content = X_valid[tweet_colname].tolist()
    
    X_train_content_feat = vectorizer.fit_transform(X_train_content).toarray()
    X_train_content_feat = pd.DataFrame(X_train_content_feat, columns=vectorizer.get_feature_names())

    X_test_content_feat = vectorizer.transform(X_test_content).toarray()
    X_test_content_feat = pd.DataFrame(X_test_content_feat, columns=vectorizer.get_feature_names())
    
    X_valid_content_feat = vectorizer.transform(X_valid_content).toarray()
    X_valid_content_feat = pd.DataFrame(X_valid_content_feat, columns=vectorizer.get_feature_names())

    print('Train:', X_train_content_feat.shape)
    print('Test:', X_test_content_feat.shape)
    print('Valid:', X_valid_content_feat.shape)

    return X_train_content_feat, X_test_content_feat, X_valid_content_feat  

def generate_emoji_vector_for_X(X_train, X_test, X_valid, emoji_colname, max_features):
    """
    Generate vectors of emojis from tweets, from three separate input 
    vectors for X (X_train, X_test and X_valid).
    
    Parameters
    ----------
        X_train : a feature matrix X for training that includes a column containing emojis
        X_test : a feature matrix X for testing that includes a column containing emojis
        X_valid : a feature matrix X for validation that includes a column containing emojis
        emoji_colname : the name of the column that contains emojis (string)
        max_features : the number of emojis to include in the vocabulary, only considering
            the top emojis ordered by term frequency across the corpus
    
    Returns
    -------
        X_train_emoji_feat, X_test_emoji_feat, X_valid_emoji_feat

    """
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
    #                              stop_words = 'english',
                                 max_features = max_features,
    #                              ngram_range = (1, 3),
                                 token_pattern=r'[^\s]+')
    X_train_emoji = X_train[emoji_colname].tolist()
    X_test_emoji = X_test[emoji_colname].tolist()
    X_valid_emoji = X_valid[emoji_colname].tolist()
    
    X_train_emoji_feat = vectorizer.fit_transform(X_train_emoji).toarray()
    X_train_emoji_feat = pd.DataFrame(X_train_emoji_feat, columns=vectorizer.get_feature_names())

    X_test_emoji_feat = vectorizer.transform(X_test_emoji).toarray()
    X_test_emoji_feat = pd.DataFrame(X_test_emoji_feat, columns=vectorizer.get_feature_names())
    
    X_valid_emoji_feat = vectorizer.transform(X_valid_emoji).toarray()
    X_valid_emoji_feat = pd.DataFrame(X_valid_emoji_feat, columns=vectorizer.get_feature_names())

    print('Train:', X_train_emoji_feat.shape)
    print('Test:', X_test_emoji_feat.shape)
    print('Valid:', X_valid_emoji_feat.shape)

    return X_train_emoji_feat, X_test_emoji_feat, X_valid_emoji_feat  

def generate_hashtag_vector_for_X(X_train, X_test, X_valid, hashtag_colname, max_features):
    """
    Generate vectors of hashtags from tweets, from three separate input 
    vectors for X (X_train, X_test and X_valid).
    
    Parameters
    ----------
        X_train : a feature matrix X for training that includes a column containing hashtags
        X_test : a feature matrix X for testing that includes a column containing hashtags
        X_valid : a feature matrix X for validation that includes a column containing hashtags
        hashtag_colname : the name of the column that contains hashtags (string)
        max_features : the number of hashtags to include in the vocabulary, only considering
            the top hashtags ordered by term frequency across the corpus
    
    Returns
    -------
        X_train_hashtag_feat, X_test_hashtag_feat, X_valid_hashtag_feat

    """
    vectorizer = CountVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
    #                              stop_words = 'english',
                                 max_features = max_features,
    #                              ngram_range = (1, 3),
                                 token_pattern=r'[^\s]+')
    X_train_hashtag = X_train[hashtag_colname].tolist()
    X_test_hashtag = X_test[hashtag_colname].tolist()
    X_valid_hashtag = X_valid[hashtag_colname].tolist()
    
    X_train_hashtag_feat = vectorizer.fit_transform(X_train_hashtag).toarray()
    X_train_hashtag_feat = pd.DataFrame(X_train_hashtag_feat, columns=vectorizer.get_feature_names())

    X_test_hashtag_feat = vectorizer.transform(X_test_hashtag).toarray()
    X_test_hashtag_feat = pd.DataFrame(X_test_hashtag_feat, columns=vectorizer.get_feature_names())
    
    X_valid_hashtag_feat = vectorizer.transform(X_valid_hashtag).toarray()
    X_valid_hashtag_feat = pd.DataFrame(X_valid_hashtag_feat, columns=vectorizer.get_feature_names())

    print('Train:', X_train_hashtag_feat.shape)
    print('Test:', X_test_hashtag_feat.shape)
    print('Valid:', X_valid_hashtag_feat.shape)

    return X_train_hashtag_feat, X_test_hashtag_feat, X_valid_hashtag_feat  

def generate_numeric_vector_for_X(X_train, X_test, X_valid, numeric_features):
    """
    Generate vectors of numeric features from tweets, from three separate input 
    vectors for X (X_train, X_test and X_valid).
    
    Parameters
    ----------
        X_train : a feature matrix X for training
        X_test : a feature matrix X for testing
        X_valid : a feature matrix X for validation
        numeric_features : numeric features to keep (list)
    
    Returns
    -------
        X_train_numeric_feat, X_test_numeric_feat, X_valid_numeric_feat
    """
    X_train_numeric = X_train[numeric_features]
    X_test_numeric = X_test[numeric_features]
    X_valid_numeric = X_valid[numeric_features]
    
    scaler = StandardScaler()
    X_train_numeric_feat = pd.DataFrame(scaler.fit_transform(X_train_numeric), 
                                        columns=X_train_numeric.columns)
    X_test_numeric_feat = pd.DataFrame(scaler.transform(X_test_numeric), 
                                        columns=X_test_numeric.columns)
    X_valid_numeric_feat = pd.DataFrame(scaler.transform(X_valid_numeric), 
                                        columns=X_valid_numeric.columns)
    print('Train:', X_train_numeric_feat.shape)
    print('Test:', X_test_numeric_feat.shape)
    print('Valid:', X_valid_numeric_feat.shape)
    
    return X_train_numeric_feat, X_test_numeric_feat, X_valid_numeric_feat

def generate_combined_features_X(X_train, X_test, X_valid):
    """
    Combine vectors of words, emojis, hashtags and numeric features 
    into one dataframe of predictors for modelling. 

    Returns
    -------
        X_train_combo, X_test_combo, X_valid_combo

    """
    # content
    X_train_content_feat, X_test_content_feat, X_valid_content_feat = generate_word_vector_for_X(
                X_train, 
                X_test, 
                X_valid, 
                tweet_colname='content_tokenized_lemma_joined',
                max_features=300
    )
    
    # emoji
    X_train_emoji_feat, X_test_emoji_feat, X_valid_emoji_feat = generate_emoji_vector_for_X(
                X_train, 
                X_test, 
                X_valid, 
                emoji_colname='emojis_joined',
                max_features=200
    )
    
    # hashtag
    X_train_hashtag_feat, X_test_hashtag_feat, X_valid_hashtag_feat = generate_hashtag_vector_for_X(
                X_train, 
                X_test, 
                X_valid, 
                hashtag_colname='hashtags_joined',
                max_features=300
    )
    
    # numeric
    numeric_features = [
     'urls_count',
     'mentions_count',
     'hashtags_count',
     'reserved_words_count',
     'emojis_count',
     'word_count',
     'pct_upper',
     'exclams_count',
     'ques_marks_count',
     'dollar_marks_count',
     'pics_count',
    ]
    
    X_train_numeric_feat, X_test_numeric_feat, X_valid_numeric_feat = generate_numeric_vector_for_X(
                X_train, 
                X_test, 
                X_valid, 
                numeric_features=numeric_features
    )
    
    # concat train, text and valid together
    print('---X_train report---')
    print(X_train_content_feat.shape)
    print(X_train_emoji_feat.shape)
    print(X_train_hashtag_feat.shape)
    print(X_train_numeric_feat.shape)
    X_train_combo = pd.concat([X_train_content_feat, X_train_emoji_feat,
                              X_train_hashtag_feat, X_train_numeric_feat], axis=1, sort=False)
    print('Combined shape:', X_train_combo.shape)
    
    print('---X_test report---')
    print(X_test_content_feat.shape)
    print(X_test_emoji_feat.shape)
    print(X_test_hashtag_feat.shape)
    print(X_test_numeric_feat.shape)
    X_test_combo = pd.concat([X_test_content_feat, X_test_emoji_feat,
                              X_test_hashtag_feat, X_test_numeric_feat], axis=1, sort=False)
    print('Combined shape:', X_test_combo.shape)

    print('---X_valid report---')
    print(X_valid_content_feat.shape)
    print(X_valid_emoji_feat.shape)
    print(X_valid_hashtag_feat.shape)
    print(X_valid_numeric_feat.shape)
    X_valid_combo = pd.concat([X_valid_content_feat, X_valid_emoji_feat,
                              X_valid_hashtag_feat, X_valid_numeric_feat], axis=1, sort=False)
    print('Combined shape:', X_valid_combo.shape)
    
    return X_train_combo, X_test_combo, X_valid_combo