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

# functions for feature engineering

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
    print('Original shape:', df.shape)
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

    print('New shape:', df.shape)
