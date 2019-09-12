from datetime import datetime
import json
import os
import pickle
import re
import sys
import time

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk import FreqDist, word_tokenize
from nltk.collocations import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import pandas as pd
import preprocessor as p
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
# from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.utils.multiclass import unique_labels
# from wordcloud import WordCloud
# from xgboost import XGBClassifier


def add_column_from_regex(df, pattern, old_col, new_colname, strip=False):
    """
    Create a new column to add to a dataframe, by applying a regex pattern 
    to an existing column.
    
    Parameters
    ----------
        df : a dataframe of tweets
        pattern : a regex pattern, used to create the new column (formatted string)
        old_col : the name of the old column (string)
        new_colname : the name of the new column to add (string)
    
    Returns
    -------
        None. 
    """
    pattern = re.compile(pattern)
    df[new_colname] = df[old_col].apply(lambda x: re.findall(pattern, x))

    if strip == True:
        df[old_col] = df[old_col].apply(lambda x: re.sub(pattern, '', x))

def add_count_column(df, old_col):
    """
    Create a new column that counts the number of instances in another column.
    
    Parameters
    ----------
        df : a dataframe of tweets
        old_col : the name of the old column (string)
        new_colname : the name of the new column to add (string)
    
    Returns
    -------
        None. 

    """
    new_colname = old_col + '_count'
    df[new_colname] = df[old_col].apply(lambda x: len(x))



# functions for parsing text from tweets

def parse_urls(tweet):
    """
    Parses a tweet for URLs.
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed URLs (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.urls == None:
        return []
    else:
        return [x.match for x in parsed_tweet.urls]

def parse_mentions(tweet):
    """
    Parses a tweet for mentions.
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed mentions (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.mentions == None:
        return []
    else:
        return [x.match for x in parsed_tweet.mentions]

def parse_hashtags(tweet):
    """
    Parses a tweet for hashtags.
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed hashtags (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.hashtags == None:
        return []
    else:
        return [x.match for x in parsed_tweet.hashtags]

def parse_reserved_words(tweet):
    """
    Parses a tweet for reserved words ("RT" or "FAV").
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed reserved words (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.reserved == None:
        return []
    else:
        return [x.match for x in parsed_tweet.reserved]

def parse_emojis(tweet):
    """
    Parses a tweet for emojis.
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed emojis (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.emojis == None:
        return []
    else:
        return [x.match for x in parsed_tweet.emojis]

def parse_smileys(tweet):
    """
    Parses a tweet for smiley faces.
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed smiley faces (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.smileys == None:
        return []
    else:
        return [x.match for x in parsed_tweet.smileys]

def parse_numbers(tweet):
    """
    Parses a tweet for numbers.
    
    Parameters
    ----------
        tweet : the text of a tweet.
        
    Returns
        A list of parsed numbers (or an empty list).
    """
    parsed_tweet = p.parse(tweet)
    if parsed_tweet.numbers == None:
        return []
    else:
        return [x.match for x in parsed_tweet.numbers]



# functions for processing text from tweets

def remove_punctuation(tweet):
    """
    Removes punctuation from the text of a tweet.
    """
    pattern = re.compile(r'[^\w\s]')  
    clean_tweet =  re.sub(pattern, '', tweet).strip()
    return clean_tweet

def calc_pct_upper(tweet):
    """
    Calculates the percantage of a tweet's uppercase letters. 
    """
    clean_text = re.sub(r"\s+", "", tweet)  # remove whitespace
    numer = sum(1 for c in clean_text if c.isupper())  # number of upper chars
    denom = len(clean_text)  # number of total chars
    if denom != 0:
        return numer/denom
    else: 
        return 0

def process_tweet(tweet):
    """
    Process the text of a tweet, by the text to lowercase, 
    stripping leading and whitespace, tokenizing and removing stop words. 

    Parameters
    ----------
        tweet : the content (text) of a tweet (string)

    Returns
    -------
        clean_tweet : the cleaned content (text) of a tweet
    """
    stop_words = set(stopwords.words('english'))
    clean_tweet = tweet.lower().strip()  # lowercase, strip whitespace
    clean_tweet = word_tokenize(clean_tweet)  # tokenize
    clean_tweet = [w for w in clean_tweet if not w in stop_words]  # remove stop words
    return clean_tweet

def get_wordnet_pos(treebank_tag):
    """
    Returns the WordNet part of speech for each Tree Bank tag.
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 

def lemmatize_tokens(tweet_tokens):
    """
    Lemmatizes the (already) tokenized text of a tweet.

    Parameters
    ----------
        tweet_tokens : the tokenized and cleaned text of a tweet (list)

    Returns:
        lem_result : a list of lemmatized tokens
    """
    lemmatizer = WordNetLemmatizer()
    tags = nltk.pos_tag(tweet_tokens)
    tags_word_net = [get_wordnet_pos(w[1]) for w in tags]
    lem_result = []  
    for i in range(len(tags_word_net)):
        if tags_word_net[i]:  
            lem_result.append(lemmatizer.lemmatize(tags[i][0],tags_word_net[i]))
        else:
            lem_result.append(tags[i][0])
    return lem_result

def add_clean_text_numeric_and_regex_features(df):
    """
    Appends numeric features and features created using regular expressions
    to a dataframe of tweet text, as well as cleaned-up tweet text. 
    
    Parameters
    ----------
        df : a dataframe containing a column labeled "content", 
             containing the content of a tweet
    
    Returns
    -------
        None. Prints shapes of the original and new dataframes. 
    """
#     print('Original shape:', df.shape)
    # add features using twitter-preprocessor
    df['urls'] = df.content.apply(parse_urls)
    df['mentions'] = df.content.apply(parse_mentions)
    df['hashtags'] = df.content.apply(parse_hashtags)
    df['reserved_words'] = df.content.apply(parse_reserved_words)
    df['emojis'] = df.content.apply(parse_emojis)
    # df['numbers'] = df.content.apply(parse_numbers)
    
    # add count features
    add_count_column(df=df, old_col='urls')
    add_count_column(df=df, old_col='mentions')
    add_count_column(df=df, old_col='hashtags')
    add_count_column(df=df, old_col='reserved_words')
    add_count_column(df=df, old_col='emojis')
    # add_count_column(df=df, old_col='numbers')
    
    # clean tweet content
    df['content_clean'] = df.content.apply(lambda x: p.clean(x))
    df['content_clean'] = df.content_clean.apply(remove_punctuation)
    
    # add feature for count of words in a tweet
    df['word_count'] = df.content_clean.apply(lambda x: len(word_tokenize(x)))
    
    # add feature for percent of characters that are uppercase
    df['pct_upper'] = df.content_clean.apply(calc_pct_upper)
    
    # add features by regex (from 'content_clean')
    add_column_from_regex(df=df, pattern=r'(\!)', 
                          old_col='content_clean', new_colname='exclams')
    add_column_from_regex(df=df, pattern=r'(\?)',
                          old_col='content_clean', new_colname='ques_marks')
    add_column_from_regex(df=df, pattern=r'(\$)',
                          old_col='content_clean', new_colname='dollar_marks')
    # add_column_from_regex(df=df, pattern=r'\d+', 
    #                       old_col='content_clean', new_colname='number', strip=True)

    # add features by regex (from 'content')
    add_column_from_regex(df=df, pattern=r'pic.twitter.com[^\s]+', 
                          old_col='content', new_colname='pics', strip=True)
    
    # add count features
    add_count_column(df=df, old_col='exclams')
    add_count_column(df=df, old_col='ques_marks')
    add_count_column(df=df, old_col='dollar_marks')
    # add_count_column(df=df, old_col='number')
    add_count_column(df=df, old_col='pics')
    
    # process tweet content
    df['content_tokenized'] = df.content_clean.apply(process_tweet)

    # lemmatize tweet tokens
    df['content_tokenized_lemma'] = df.content_tokenized.apply(lemmatize_tokens)
    
    # join tokens back into a string
    df['content_tokenized_lemma_joined'] = df.content_tokenized_lemma.apply(lambda x: " ".join(x))
    df.content_tokenized_lemma_joined = df.content_tokenized_lemma_joined.fillna(value='')

    df['emojis_joined'] = df.emojis.apply(lambda x: " ".join(x))
    df['hashtags_joined'] = df.hashtags.apply(lambda x: " ".join(x))

    # filter out columns i don't need
    df = df[[
     'author', 
     'content', 
     'publish_date',
     'target',
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
     'content_tokenized_lemma_joined',
     'emojis_joined',
     'hashtags_joined'
    ]]
    
#     print('New shape:', df.shape)
    return df



def generate_word_vector_for_new_tweet(df, tweet_colname, max_features, stop_words, vocabulary):
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
          blah blah blah

    """
    vectorizer = TfidfVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 max_features = max_features,
                                 stop_words = stop_words,
                                 ngram_range = (1, 3),
                                 vocabulary=vocabulary)
    df_content = df[tweet_colname].tolist()
    
    df_content_feat = vectorizer.fit_transform(df_content).toarray()
    df_content_feat = pd.DataFrame(df_content_feat, columns=vectorizer.get_feature_names())
    
#     print('Word vector shape: ', df_content_feat.shape)
    return df_content_feat
  
  
  
def generate_emoji_vector_for_new_tweet(df, emoji_colname, max_features, vocabulary):
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
         blah blah blah

    """
    vectorizer = TfidfVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
    #                              stop_words = 'english',
                                 max_features = max_features,
    #                              ngram_range = (1, 3),
                                 token_pattern=r'[^\s]+',
                                 vocabulary=vocabulary)
    df_emoji = df[emoji_colname].tolist()
    
    df_emoji_feat = vectorizer.fit_transform(df_emoji).toarray()
    df_emoji_feat = pd.DataFrame(df_emoji_feat, columns=vectorizer.get_feature_names())
    
#     print('Emoji vector shape:', df_emoji_feat.shape)
    return df_emoji_feat

  
  
def generate_hashtag_vector_for_new_tweet(df, hashtag_colname, max_features, vocabulary):
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
         blah blah blah

    """
    vectorizer = TfidfVectorizer(analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
    #                              stop_words = 'english',
                                 max_features = max_features,
    #                              ngram_range = (1, 3),
                                 token_pattern=r'[^\s]+',
                                 vocabulary=vocabulary)
    df_hashtag = df[hashtag_colname].tolist()
    
    df_hashtag_feat = vectorizer.fit_transform(df_hashtag).toarray()
    df_hashtag_feat = pd.DataFrame(df_hashtag_feat, columns=vectorizer.get_feature_names())
    
#     print('Hashtag vector shape:', df_hashtag_feat.shape)
    return df_hashtag_feat
  
  
  
def generate_numeric_vector_for_new_tweet(df, numeric_features):
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
        blah blah blah
    """
    df_numeric = df[numeric_features]
    
    scaler = StandardScaler()
    df_numeric_feat = pd.DataFrame(scaler.fit_transform(df_numeric), 
                                        columns=df_numeric.columns)
    
#     print('Numeric vector shape:', df_numeric_feat.shape)
    return df_numeric_feat
  
  
  
def generate_combined_features_new_tweet(df, vocabulary):
    """
    Combine vectors of words, emojis, hashtags and numeric features 
    into one dataframe of predictors for modelling. 

    Returns
    -------
        blah blah blah

    """
    word_vocab = vocab[:300]
    emoji_vocab = vocab[300:500]
    hashtag_vocab = vocab[500:800]
    
    # content
    df_content_feat = generate_word_vector_for_new_tweet(
                df,
                tweet_colname='content_tokenized_lemma_joined',
                max_features=300,
                stop_words=['american', 'obamas', 'supporter', 'thanks'],
                vocabulary=word_vocab
    )
    
    # emoji
    df_emoji_feat = generate_emoji_vector_for_new_tweet(
                df,
                emoji_colname='emojis_joined',
                max_features=200,
                vocabulary=emoji_vocab
    )    
    
    # hashtag
    df_hashtag_feat = generate_hashtag_vector_for_new_tweet(
                df,
                hashtag_colname='hashtags_joined',
                max_features=300,
                vocabulary=hashtag_vocab
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
    
    df_numeric_feat = generate_numeric_vector_for_new_tweet(
                df,
                numeric_features=numeric_features
    )
    
    # concat train, text and valid together
#     print('---Combined vector---')
#     print(df_content_feat.shape)
#     print(df_emoji_feat.shape)
#     print(df_hashtag_feat.shape)
#     print(df_numeric_feat.shape)
    all_features = pd.concat([df_content_feat, df_emoji_feat, df_hashtag_feat, df_numeric_feat], axis=1)
#     print('Final shape:', all_features.shape)
    
    return all_features



def make_prediction_on_new_tweet(text=str):
    """
    """
    # import training vocabulary and RF model
    with open('drive/My Drive/data/training_vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    best_rf = joblib.load('drive/My Drive/data/best_rf_all_data_20190906.pkl')
    
    # convert new tweet into dataframe
    new_tweet = pd.DataFrame({'content': [text],
                              'target': None,
                              'author': None,
                              'publish_date': None})
    new_tweet_cleaned = add_clean_text_numeric_and_regex_features(new_tweet)
    new_tweet_all_features = generate_combined_features_new_tweet(new_tweet_cleaned, vocabulary=vocab)
    
    # make new predictions
    pred_class = best_rf.predict(new_tweet_all_features)[0]
    pred_proba = best_rf.predict_proba(new_tweet_all_features)[0][1]
    
    print('Predicted class: ', pred_class)
    print('Predicted proba: ', pred_proba)
    return pred_class, pred_proba