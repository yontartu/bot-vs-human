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

# functions for feature engineering

def add_column_from_regex(df, pattern, old_col, new_colname):
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


