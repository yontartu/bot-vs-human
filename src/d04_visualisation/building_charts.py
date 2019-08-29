import pandas as pd
import numpy as np
import os
import re
import json
from datetime import datetime
import pickle
import time


# functions to generate dataframes for visuals

def generate_hashtags_counts(df, new_filepath):
    """
    Generates a dataframe of all hashtags and corresponding counts,
    ordered by most to least instances.
    
    Parameters
    ----------
        df : a dataframe containing a column for 'hashtags' (a list of hashtags)
             and another column for 'hashtags_count' (the number of hashtags)
        new_filepath : the name and path to save the dataframe with hashtag counts
        
    Returns
    -------
        None. Prints a progress report and the filepath to the new file.
    """
    print('Processing all hashtags into one list...')
    all_hashtags = df[df.hashtags_count > 0].hashtags.sum()
    print('...done.')
    print('List of all hashtags has length', len(all_hashtags))
    top_hashtags = pd.DataFrame.from_dict(Counter(all_hashtags), orient='index').reset_index()
    top_hashtags.columns = ['hashtag', 'counts']
    top_hashtags.sort_values('counts', ascending=False, inplace=True)
    top_hashtags = top_hashtags.reset_index(drop=True)
    top_hashtags.to_csv(new_filepath, index=False)
    print('  Saved', new_filepath, 'with shape', top_hashtags.shape)

def generate_tweets_by_date(df, new_filepath):
    """
    Generates a dataframe of counts of tweets by date, ordered by date.
    
    Parameters
    ----------
        df : a dataframe containing a column for 'publish_date' (datetime type)
        new_filepath : the name and path to save the dataframe with hashtag counts
        
    Returns
    -------
        None. Prints a progress report and the filepath to the new file.
   
    """
    tweets_by_date = df.groupby(df.publish_date.apply(lambda x: x.date())).size().reset_index()
    tweets_by_date.columns = ['publish_date', 'num_of_tweets']
    tweets_by_date.sort_values('publish_date', inplace=True)
    tweets_by_date.to_csv(new_filepath, index=False)
    print('  Saved', new_filepath, 'with shape', tweets_by_date.shape)
