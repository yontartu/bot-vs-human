# Classifying Tweets from Russian Troll Accounts

My project involves building a machine learning model to detect tweets that resemble those from Russian troll twitter accounts. 

If you want to explore some of my tweet-level predictions, check out [this tool](https://joeygoodman.us/projects/1-russian-trolls). You can also use [this web app](https://russian-troll-detector.herokuapp.com/) to make predictions on new tweets.

![alt text](https://github.com/yontartu/bot-vs-human/blob/master/results/img/explore_tweet_predictions.gif)

![alt text](https://github.com/yontartu/bot-vs-human/blob/master/results/img/web_app_demo.gif)

## Set-up Instructions

#### 1. Collect data

My project required a dataset of tweets, labeled by whether they were sent from a Russian troll account or not. The data on Russian troll tweets was combiled by Linvill and Warren of Clemson University, and open-sourced through [FiveThirtyEight](https://github.com/fivethirtyeight/russian-troll-tweets). In order to construct a control group, I employed the `twint` package to collect a sample of tweets from verified users, tweeting within the same time period as the Russian troll accounts. The command I used to collect this data was:

```bash
$ bash collect_verified_tweets.sh
```

#### 2. Process 

