# import requests
# from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import re
import json
from datetime import datetime
import pickle
import time

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


# functions to merge/clean/process raw data 

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

def create_combined_right_troll_dataframe(new_filepath):
    """
    Create a CSV file containing all tweets from accounts labeled as "Right
    Trolls" by academic researchers. 
    
    Parameters
    ----------
        new_filepath : the name and path to save the new combined dataframe
        
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

    right_trolls_all.to_csv(new_filepath, index=False)
    print('  Saved', new_filepath, 'with shape', right_trolls_all.shape)


# functions to 