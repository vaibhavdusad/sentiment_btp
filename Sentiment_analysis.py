
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
import tweepy
from time import sleep
from datetime import datetime
import nltk
from datetime import timedelta
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
ps = PorterStemmer()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


# In[2]:


consumer_key = 'X7kyc3HCZPlM1Vjk7jt0EFKwg'
consumer_secret = 'DStvgODt3tuzeIY2nPnnuVYeux8XPsVpEyunuDfPohuQo8qSqp'
access_token = '527534130-OgdG8JZDojdMEst0D0cOx0zaVX7PQ8d5Oov0UkVA'
access_token_secret = '3xVdXJsBTufuQS4kMnOnHbH374AyrKVBJfEhTlp5Tgpfn'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


# In[3]:


sentiment=pd.read_csv('vaibhav_sentiment.csv',encoding='latin-1')
sent_dict={}
words=[]
for index,row in sentiment.iterrows():
    sent_dict[row['word']]=row['number']
    words.append(row['word'])


# In[4]:


def get_sentiment(token):
    tokenizer = RegexpTokenizer(r'\w+')
    sum=0
    m=tokenizer.tokenize(token)
    n=[l for l in m if not l in nltk.corpus.stopwords.words('English')]
    m=[w for w in n if w in words]
    i=0
    for c in m:
        i=i+1
        g=int(sent_dict[c])
        sum+=g
    if i>0:
        sumg=int(sum/i)
    else:
        sumg=0
    return sumg


# In[5]:


def getData(keyword,numberoftweets,last_tweet):
    data2={}
    sentimentl=[]
    time=datetime.strptime(last_tweet,'%Y-%m-%d %H:%M')
    number=1
    for tweet in tweepy.Cursor(api.search, q=keyword, lang="en", result_type='recent').items(numberoftweets):
        try:
            line=tweet.text
            if not tweet.retweeted and not tweet.text.startswith('RT'):
                eastern_time = tweet.created_at + timedelta(minutes=30)+ timedelta(hours=5)
                if eastern_time>time:
                    edt_time = eastern_time.strftime('%Y-%m-%d %H:%M')
                    data2[edt_time]=line.lower()
                else:
                    continue
            else:
                continue
        except tweepy.TweepError as e:
            print(e.reason)
        except StopIteration:
            break
    z=pd.Series(data2,name='Tweet')
    z.index.name='Timestamp'
    stockn=z.reset_index()    
    return stockn


# In[15]:


stocks= pd.read_csv('Stock_names.csv',header=None)


# In[7]:


end_time=[]
for index,row in stocks.iterrows():
    peace=pd.DataFrame([],columns=['Tweet','Timestamp','score'])
    for i in range(1,len(row)-1):
        try:
            if not np.isnan(row[i]):
                x=getData(row[i],20,row[5])  
                peace=pd.concat([peace,x],ignore_index=True)
            else:
                break
        except TypeError:
            x=getData(row[i],20,row['6'])  
            peace=pd.concat([peace,x],ignore_index=True)
    for index2,row2 in peace.iterrows():
                row2['score']=get_sentiment(row2['Tweet'])
                row2['Tweet']=row2['Tweet'].encode('utf-8')
    peace.sort_values(by='Timestamp',inplace=True)
    last_time=peace['Timestamp'][len(peace)-1]
    end_time.append(last_time)
    file_name=row['1']+".csv"    
    with open(file_name, 'a') as f:
            peace.to_csv(f, header=False,encoding='utf-8')


# In[10]:


end_df=pd.DataFrame({'last_time':end_time})  
stocks_final=pd.concat([stocks,end_df],axis=1,ignore_index=True)
with open('Stock_names.csv', 'w') as g:
            stocks.to_csv(g, header=False,encoding='utf-8',index_label=False)


# In[90]:


for name in stocks[0]:
    filename=name+'.csv'
    stock_name=pd.read_csv(filename,header=None)  
    day=[]
    for dates in stock_name[1]:
        dates=datetime.strptime(dates,'%Y-%m-%d %H:%M')
        newdate=dates.date()
        day.append(newdate)
    stock_name['day']=day
    daygroup=stock_name.groupby('day',as_index=False,sort=False)[3].mean()
    dates = matplotlib.dates.date2num(daygroup['day'])
    fig = plt.figure()
    plt.title(name)
    ax = fig.add_subplot(1,1,1)  
    plt.xticks(rotation=70)
    plt.plot_date(dates, daygroup[3])
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) 
    plt.show()

