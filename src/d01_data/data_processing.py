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
import numpy as np
import pandas as pd
import seaborn as sns

# functions to work with pickle files 

def save_pickle(obj, filename):
	"""
	Save an object as a .pkl file, with specified filename.
	The filename must be a string.
	"""
	with open(filename, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
		print('Saved', filename)

def load_pickle(name):
	"""
	Load a .pkl file from a specific path.
	The name must be a string.
	"""
	with open(name, 'rb') as f:
		return pickle.load(f)



# functions to merge/clean/process russian troll data

def filter_dataframe_to_right_trolls(filepath, new_filepath, chunksize):
    """
    Creates a subset of Russian Twitter bot tweets for a single dataframe, 
    filtering tweets by those labeled as from a "RightTroll" account
    by academic researchers.
    
    Parameters
    ----------
        filepath : path to read in a CSV of Russian bot tweets
        new_filepath: the name and path to save the new filtered dataframe
        chunksize : the number of rows to read in at one time
        
    Returns
    -------
        None. Prints a progress report after processing each file. 
    """
    df_to_save = pd.DataFrame()
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunksize)):
        print(f'    Processing chunk {i+1}')
        new_df = chunk[chunk.account_category == 'RightTroll']
        df_to_save = pd.concat([df_to_save, new_df], axis=0, sort=False)
        print('    ', df_to_save.shape)
    
    df_to_save.to_csv(new_filepath, index=False)
    print('    Saved', new_filepath, 'with shape', df_to_save.shape)

def process_files_into_right_trolls(num_of_files, chunksize):
    """
    Process all raw CSV datafiles into CSVs solely containing tweets from 
    "Right Trolls."
    
    Parameters
    ----------
        num_of_files : the number of CSVs to process
        chunksize: the number of rows to read in at one time
        
    Returns
    -------
        None. Prints a progress report after processing each file.
        
    """
    for i in range(1, num_of_files+1):
        filepath = f'../../data/01_raw/russian_troll_tweets/IRAhandle_tweets_{i}.csv'
        print(f'Reading in {filepath}')
        new_filepath = f'../../data/02_intermediate/right_trolls_{i}.csv'
        filter_dataframe_to_right_trolls(filepath, new_filepath, chunksize=chunksize)

def create_combined_right_troll_csv(new_filepath):
    """
    Create a CSV file containing all tweets from accounts labeled as "Right
    Trolls" by academic researchers. 
    
    Parameters
    ----------
        new_filepath : the name and path to save the new combined CSV
        
    Returns
    -------
        None. Prints a progress report after processing each file.
    """
    right_trolls_all = pd.DataFrame()
    for i in range(1, 14):
        filepath = f'../../data/02_intermediate/right_trolls_{i}.csv'
        print(f'Reading in {filepath}') 
        new_df = pd.read_csv(filepath)
        right_trolls_all = pd.concat([right_trolls_all, new_df], axis=0, sort=False)
        del(new_df)

    right_trolls_all['target'] = 1
    right_trolls_all.to_csv(new_filepath, index=False)
    print('  Saved', new_filepath, 'with shape', right_trolls_all.shape)

def trim_right_trolls_csv(orig_filepath, new_filepath):
    """
    Creates a subset of a `right_trolls` CSV, reducing the number
    of features and rows.
    
    Parameters
    ----------
        orig_filepath : path to `right_trolls` CSV
        new_filepath : the name and path to save the trimmed dataframe
    
    Returns
    -------
        None. Prints a progress report and the filepath to the new file.
    """
    df = pd.read_csv(orig_filepath)
    start_shape = df.shape
    print('Original shape', start_shape)
    
    # convert date to datetime
    # df['publish_date'] = pd.to_datetime(df.publish_date)
    
    # filter out retweets
    df = df[df.retweet == 0]
    df = df[df.content.str.startswith('RT ') == False]
    print('Filtered out', start_shape[0] - df.shape[0], 'rows (retweets)')
    intermed_shape = df.shape
    
    # filter to English tweets
    df = df[df.language == 'English']
    print('Filtered out', intermed_shape[0] - df.shape[0], 'rows (non-English)')
    
    # filter down to features of interest
    features_to_keep = ['author', 'content', 'publish_date', 'target']
    df = df[features_to_keep]
    intermed_shape = df.shape

    # drop duplicate tweets
    df.drop_duplicates(subset='content', inplace=True)
    print('Filtered out', intermed_shape[0] - df.shape[0], 'rows (duplicate tweets)')
    
    df.to_csv(new_filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
    print('  Saved', new_filepath, 'with shape', df.shape)



# functions to merge/clean/process verified tweets data

def create_verified_tweets_csv(new_filepath):
    """
    Create a CSV file containing all verified tweets collected using the Twint package.
    Combines datafiles from individual Twint queries (both hashtag-specific and not).
    
    Parameters
    ----------
        new_filepath : the name and path to save the new combined CSV
        
    Returns
    -------
        None. Prints a progress report after processing each file.    
    """
    verified_tweets = pd.read_csv('../../data/01_raw/verified_tweets/twint_query_NOHASHTAG.csv')
    print('Shape after loading tweets not pulled using hashtags', verified_tweets.shape)
    twint_files_hashtag = os.listdir('../../data/01_raw/verified_tweets/')
    twint_files_hashtag.remove('old_files')
    twint_files_hashtag.remove('twint_query_NOHASHTAG.csv')
    for filename in twint_files_hashtag:
        print('Processing', filename)
        path = '../../data/01_raw/verified_tweets/' + filename
        new_df = pd.read_csv(path)
        verified_tweets = pd.concat([verified_tweets, new_df], axis=0, sort=False)
    start_shape = verified_tweets.shape
    print('Original shape (after processing all files):', start_shape)

    # drop rows with duplicated tweet content
    verified_tweets.drop_duplicates(subset='tweet', inplace=True)
    print('Filtered out', start_shape[0] - verified_tweets.shape[0], 'rows (duplicate tweets)')

    # add target column
    verified_tweets['target'] = 0
    
    # add datetime column
    verified_tweets['publish_date'] = pd.to_datetime(verified_tweets['date'] + ' ' + verified_tweets['time'])

    # filter down to columns of interest
    features_to_keep = ['username', 'tweet', 'publish_date', 'target']
    verified_tweets = verified_tweets[features_to_keep]
    
    # rename columns (to match russian troll tweets)
    verified_tweets.columns = ['author', 'content', 'publish_date', 'target']

    verified_tweets.to_csv(new_filepath, index=False)
    print('  Saved', new_filepath, 'with shape', verified_tweets.shape)



# function to combine right troll tweets with verified tweets (downsampling troll tweets)

def create_tweets_all_csv(path_to_right_trolls, path_to_verified_tweets, new_filepath, random_state):
    """
    Create a merged CSV combining Right Troll tweets as well as tweets from verified users.
    First downsample from Right Troll tweets, so that its shape matches that of verified tweets.
    The two CSVs must have been preprocessed so that their columns align.

    Parameters
    ----------
        path_to_right_trolls : a CSV containing right troll tweets, and four columns for 
            "author", "content", "publish_date" and "target"
        path_to_verified_tweets : a CSV containing tweets from verified users, and four columns for 
            "author", "content", "publish_date" and "target"
        new_filepath : the name and path to save the new combined CSV
        random_state : random seed to generate sample of right troll tweets
                    
    Returns
    -------
        None. Prints a progress report after processing each file.    

    """
    right_trolls = pd.read_csv(path_to_right_trolls, parse_dates=[2])
    print('Shape of right_trolls:', right_trolls.shape)
    verified_tweets = pd.read_csv(path_to_verified_tweets, parse_dates=[2])
    print('Shape of verified_tweets:', verified_tweets.shape)

    # downsample from right trolls (shape should match verified_tweets)
    right_trolls_sample = right_trolls.sample(n=verified_tweets.shape[0], random_state=random_state)
    print('Shape of right_trolls_sample:', right_trolls_sample.shape)

    # merge dataframes
    tweets_all = pd.concat([right_trolls_sample, verified_tweets], axis=0, sort=False)
    tweets_all = tweets_all.reset_index(drop=True)
    tweets_all.to_csv(new_filepath, index=False)
    print('  Saved', new_filepath, 'with shape', tweets_all.shape)

