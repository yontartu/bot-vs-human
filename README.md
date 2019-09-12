# Classifying Tweets from Russian Troll Accounts

My project involves building a machine learning model to detect tweets from Russian troll twitter accounts. 

If you want to explore some of my tweet-level predictions, check out [this tool](https://joeygoodman.us/projects/1-russian-trolls). You can also use [this web app](https://russian-troll-detector.herokuapp.com/) to make predictions on new tweets.

![alt text](https://github.com/yontartu/bot-vs-human/blob/master/results/img/explore_tweet_predictions.gif)

![alt text](https://github.com/yontartu/bot-vs-human/blob/master/results/img/web_app_demo.gif)

## Research Process

#### 1. Collect data

For this project, I construct a dataset of about 340,000 labeled tweets, denoting whether they were sent from a Russian troll account or not. The data on Russian troll tweets was combiled by Linvill and Warren of Clemson University, and open-sourced through [FiveThirtyEight](https://github.com/fivethirtyeight/russian-troll-tweets). In order to construct a control group, I employed the [`twint`](https://github.com/twintproject/twint) package to collect a sample of tweets from verified users, tweeting within the same time period as the Russian troll accounts. This collection can be replicated using this code:

```bash
$ bash collect_verified_tweets.sh
```

#### 2. Process text data and feature engineering

I perform extensive preprocessing of the raw text from tweets (including tokenization, stop word removal, and lemmatization), as well as feature engineering. You can find these functions in `src`:

- Data processing: [`src/d01_data/data_processing.py`](https://github.com/yontartu/bot-vs-human/blob/master/src/d01_data/data_processing.py)
- Text processing: [`src/d02_features/text_preprocessing.py`](https://github.com/yontartu/bot-vs-human/blob/master/src/d02_features/text_preprocessing.py)

#### 3. Build models and tune hyperparameters

I test out several different classifiers, and settle on random forest as the best estimator. Here's a summary comparing the performance of various baseline models I tested:

![alt text](https://github.com/yontartu/bot-vs-human/blob/master/results/img/04_model_comparison.png)

You can take a look at my entire workflow in a single notebook, which lives [here](https://github.com/yontartu/bot-vs-human/blob/master/notebooks/04_reports/presentation_notebook.ipynb). 

#### 4. Results

After hyperparameter tuning through grid search, my random forest classifier achieves a recall score of **93.3%**.

You can explore tweet predictions using [this tool](https://joeygoodman.us/projects/1-russian-trolls).

You can try making predictions on new tweets using [this web app](https://russian-troll-detector.herokuapp.com/). (The source code for this web app is available [here](https://github.com/yontartu/russian-troll-detector-webapp))

A slide deck summarizing my project can be found [here](https://github.com/yontartu/bot-vs-human/blob/master/results/JoeyGoodman_FinalPresentation.pdf).