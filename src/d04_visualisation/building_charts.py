from collections import Counter
from datetime import datetime
import json
import jsonpickle
import os
import pickle
import re
import sys
import time

from bokeh.io import export_png
import bokeh.models as bmo
from bokeh.models import BoxSelectTool, DatetimeTickFormatter
import bokeh.plotting as bpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

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

def generate_data_for_wordclouds(df):
    """
    Generate input data for word clouds.
    
    Parameters
    ----------
        df : a dataframe of all tweets (must contain a field called "target")
        
    Returns
    -------
        text_all_trolls, text_all_verified
    """
    words_all_trolls = tweets_all[tweets_all.target == 1]
    words_all_verified = tweets_all[tweets_all.target == 0]
    text_all_trolls = " ".join(words_all_trolls.content_tokenized_lemma_joined.tolist())
    text_all_verified = " ".join(words_all_verified.content_tokenized_lemma_joined.tolist())
    
    return text_all_trolls, text_all_verified


def code_troll_column(x):
    """
    Small helper function for generate_data_for_bokeh()
    """
    if x == 0:
      return 'No'
    else:
      return 'Yes'

def code_color_column(x):
    """
    Small helper function for generate_data_for_bokeh()
    """
    if x == 0:
      return 'turquoise'
    else:
      return 'salmon'

def generate_data_for_bokeh(path_to_data, new_filepath):
    """
    Generate input data for a bokeh visualization exploring tweet predictions.
    
    Parameters
    ----------
        path_to_data : path to CSV containing tweet-level predictions
        new_filepath : the name and path to save the processed data
    
    Returns
    -------
        None. Prints a report.
    """
    df = pd.read_csv(path_to_data, parse_dates=[3])
    print('Original shape: ', df.shape)
    df.author = df.author.apply(lambda x: '@'+x)
    df.publish_date = df.publish_date.apply(lambda x: datetime.date(x))
    df['troll'] = df.target.apply(code_troll_column)
    df['color'] = df.target.apply(code_color_column)
    df = df[['author', 'content', 'publish_date', 'y_pred_proba', 'troll', 'color']]
    df.columns = ['username', 'tweet_text', 'datetime', 'pred_proba', 'troll', 'color']
    df.to_csv(new_filepath)
    print('New shape: ', df.shape)
    print('Saved', new_filepath)



# functions to generate plots

def plot_word_cloud(data, colormap, new_filepath):
    """
    Plot a word cloud with a specified colormap.
    
    Parameters
    ----------
        data : text data to use in the word cloud (string)
        colormap : the color scheme to use in the word cloud
        new_filepath : the name and path of save the image
    
    Returns
    -------
        None. Shows the plot.
        
    """
    wordcloud = WordCloud(width=800, height=480, margin=0,
                          max_words=300, background_color="white",
                          colormap=colormap
                           ).generate(data)

    plt.figure(figsize=(8,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.savefig(new_filepath)
    print('Saved ', new_filepath)
    plt.show()


def plot_bokeh_explore_predictions(path_to_data, new_filepath):
    """
    Create a bokeh plot allowing users to explore predictions for individual tweets.

    Parameters
    ----------
        path_to_data : path to CSV containing pre-processed data for bokeh
        new_filepath : the name and path to save the HTML for the bokeh chart
    
    Returns
    -------
        None. Shows the plot.
    """
    bpl.output_notebook()

    TOOLTIPS = """
        <div style="width:300px;">
            <div>
                <span style="font-size: 12px; color: #2eafc6;">Handle: </span>
                <span style="font-size: 12px; font-weight: bold;">@username</span>
            </div>
            <div>
                <span style="font-size: 12px; color: #2eafc6;">Tweet: </span>
                <span style="font-size: 12px; ">@tweet_text</span>
            </div>
            <div>
                <span style="font-size: 12px; color: #2eafc6;">Predicted prob. Russian troll: </span>
                <span style="font-size: 12px; font-weight: bold;">@pred_proba</span>
            </div>
            <div>
                <span style="font-size: 12px; color: #2eafc6;">Actually a Russian troll? </span>
                <span style="font-size: 12px; font-weight: bold;">@troll</span>
            </div>
        </div>
    """
    data_for_bokeh = pd.read_csv(path_to_data, parse_dates=[3])
    source_troll = bpl.ColumnDataSource.from_df(data_for_bokeh[data_for_bokeh.troll == 'Yes'])
    source_not_troll = bpl.ColumnDataSource.from_df(data_for_bokeh[data_for_bokeh.troll == 'No'])
    hover = bmo.HoverTool(tooltips=TOOLTIPS)

    p = bpl.figure(tools=[hover, 'box_zoom', 'reset', 'pan'], x_axis_type='datetime',
                   x_axis_label='Date', y_axis_label='Predicted probability troll', 
                   plot_width=900, plot_height=450)

    p.circle(
        'datetime', 'pred_proba', source=source_not_troll, color='color', size=9, 
        fill_alpha=0.1, legend='Not Russian Troll')
    p.circle(
        'datetime', 'pred_proba', source=source_troll, color='color', size=9, 
        fill_alpha=0.1, legend='Russian Troll')

    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '10pt'  
    p.yaxis.major_label_text_font_size = '10pt' 
    p.xaxis.axis_label_text_font_size = '12pt' 
    p.xaxis.axis_label_text_font_style = 'bold' 
    p.yaxis.axis_label_text_font_size = '12pt' 
    p.yaxis.axis_label_text_font_style = 'bold' 

    p.xaxis.formatter=DatetimeTickFormatter(
            hours=["%b '%y"],
            days=["%b '%y"],
            months=["%b '%y"],
            years=["%b '%y"],
        )

    p.title.text = "Explore Tweet Predictions"
    p.title.align = "center"
    p.title.text_color = "Black"
    p.title.text_font_size = "20px"
    p.legend.location = "top_left"
    p.legend.click_policy="hide"

    bpl.output_file(new_filepath)
    print('Saved', new_filepath)
    bpl.show(p)


