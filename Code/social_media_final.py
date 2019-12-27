#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:44:58 2019

@author: abinavrameshsundararaman
"""

import os
os.chdir("/Users/abinavrameshsundararaman/Documents/McGill/Courses/Winter 2019/Social Media Analytics/Final Project_Social_Media")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import all libraries

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from nltk.corpus import stopwords
import nltk
from nltk.corpus import reuters
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from operator import itemgetter
from sklearn.metrics import classification_report
import csv
import os
import collections
import os, csv, lda, nltk
from nltk.tokenize import PunktSentenceTokenizer, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import ast


## All functions


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
User defined functions

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# remove non-alphabets

def remove_alphabets(lst):
    lst1=[]
    for i in lst : 
        if(i.isalpha()):
            lst1.append(i)
    return(lst1)
            
        
# Clean the words
def content_without_stopwords_lower(text):
    stopwords = nltk.corpus.stopwords.words('english')
    wnl = nltk.WordNetLemmatizer()
    ps = nltk.stem.PorterStemmer()
    #ps.stem('grows')
    content = [ps.stem(wnl.lemmatize(w.lower()) )for w in text if w.lower() not in stopwords]
    return content


### Make wordcloud

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        relative_scaling = 1.0,
        collocations=False,
        random_state=1 # chosen at random by flipping a coin; it was heads
        
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Word Cloud -- Using Hashags 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


airpower = pd.read_excel("airpower_en_v1.xlsx")
airpower.columns



#drop na. purpose: avoid erro
airpower['hashtags_clean'].dropna(inplace=True)


#tokenize words
airpower["hashtags_clean_token"] = airpower["hashtags_clean"].apply(nltk.word_tokenize)
airpower['hashtags_clean_token'].dropna(inplace=True)
airpower['hashtags_clean_token']=airpower['hashtags_clean_token'].apply(lambda x :remove_alphabets(x) )

airpower['hashtags_clean_token'].dropna(inplace=True)  
airpower["hashtags_clean_without_stop"] =airpower["hashtags_clean_token"].apply(lambda x :content_without_stopwords_lower(x) )

airpower["hashtags_clean_without_stop"].dropna(inplace=True)

tags_combined=[]
for  i in airpower["hashtags_clean_without_stop"] : 
    for j in i :
        tags_combined.append(j)
    

frequencies_tags=nltk.FreqDist(tag for tag in tags_combined  )

#show_wordcloud(airpower["hashtags_clean_without_stop"])

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=200,
    max_font_size=40, 
    scale=3,
    #relative_scaling = 1.0,
    #collocations=False,
    random_state=1 # chosen at random by flipping a coin; it was heads
    
).generate_from_frequencies(frequencies_tags)

fig = plt.figure(1, figsize=(12, 12))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Word Cloud --Before date--text

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

before_cancel=airpower[(airpower['created_at'] <'2019-03-29') ]
after_cancel=airpower[(airpower['created_at'] >='2019-03-29') ]


#drop na. purpose: avoid erro
before_cancel['text'].dropna(inplace=True)


#tokenize words
before_cancel["text_clean_token"] = before_cancel["text"].apply(nltk.word_tokenize)


before_cancel['text_clean_token'].dropna(inplace=True)
before_cancel['text_clean_token']=before_cancel['text_clean_token'].apply(lambda x :remove_alphabets(x) )

before_cancel['text_clean_token'].dropna(inplace=True)  
before_cancel["text_clean_without_stop"] =before_cancel["text_clean_token"].apply(lambda x :content_without_stopwords_lower(x) )

before_cancel["text_clean_without_stop"].dropna(inplace=True)

text_combined=[]
for  i in before_cancel["text_clean_without_stop"] : 
    for j in i :
        text_combined.append(j)
    
#temp=" ".join(text_combined) 
frequencies_text=nltk.FreqDist(tag for tag in text_combined  )

#show_wordcloud(before_cancel["text_clean_without_stop"])
#show_wordcloud(temp)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=200,
    max_font_size=40, 
    scale=3,
    #relative_scaling = 1.0,
    #collocations=False,
    random_state=1 # chosen at random by flipping a coin; it was heads
    
).generate_from_frequencies(frequencies_text)

fig = plt.figure(1, figsize=(12, 12))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Word Cloud --after cancellation date--text

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#drop na. purpose: avoid erro
after_cancel['text'].dropna(inplace=True)


#tokenize words
after_cancel["text_clean_token"] = after_cancel["text"].apply(nltk.word_tokenize)


after_cancel['text_clean_token'].dropna(inplace=True)
after_cancel['text_clean_token']=after_cancel['text_clean_token'].apply(lambda x :remove_alphabets(x) )

after_cancel['text_clean_token'].dropna(inplace=True)  
after_cancel["text_clean_without_stop"] =after_cancel["text_clean_token"].apply(lambda x :content_without_stopwords_lower(x) )

after_cancel["text_clean_without_stop"].dropna(inplace=True)

text_combined=[]
for  i in after_cancel["text_clean_without_stop"] : 
    for j in i :
        text_combined.append(j)
    
#temp=" ".join(text_combined) 
frequencies_text=nltk.FreqDist(tag for tag in text_combined  )

#show_wordcloud(after_cancel["text_clean_without_stop"])
#show_wordcloud(temp)

wordcloud = WordCloud(
    background_color='white',
    stopwords=stopwords,
    max_words=200,
    max_font_size=40, 
    scale=3,
    #relative_scaling = 1.0,
    #collocations=False,
    random_state=1 # chosen at random by flipping a coin; it was heads
    
).generate_from_frequencies(frequencies_text)

fig = plt.figure(1, figsize=(12, 12))
plt.axis('off')

plt.imshow(wordcloud)
plt.show()





"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
EMOJI ANALYSIS -- Using text--before and AFTER cancellation

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import emoji
import re

## Create the function to extract the emojis
def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux=[' '.join(r.findall(s)) for s in a_list]
    return(aux)

## Execute the function
    
#### BEFORE cancellation

before_cancel['text'].dropna(inplace=True)


before_cancel_text_list=list(before_cancel['text'])
before_cancel_emoji_list=[]

for i in before_cancel_text_list : 
    lst1=extract_emojis(i)
    for i in lst1:
        if i!="" :
           before_cancel_emoji_list.append(i) 
           
          
frequencies_emoji_before=nltk.FreqDist(tag for tag in before_cancel_emoji_list  )

df_fdist = pd.DataFrame.from_dict(frequencies_emoji_before, orient='index').reset_index()
df_fdist.columns = ['Emoji','Frequency']


#### AFTER cancellation


after_cancel['text'].dropna(inplace=True)


after_cancel_text_list=list(after_cancel['text'])
after_cancel_emoji_list=[]

for i in after_cancel_text_list : 
    lst1=extract_emojis(i)
    for i in lst1:
        if i!="" :
           after_cancel_emoji_list.append(i) 
           
frequencies_emoji_after=nltk.FreqDist(tag for tag in after_cancel_emoji_list  )

df_fdist_after = pd.DataFrame.from_dict(frequencies_emoji_after, orient='index').reset_index()
df_fdist.columns = ['Emoji','Frequency']
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Network Analysis--top influencers

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# import the dataset which was cleaned in R

clean_twitter_data = pd.read_excel("/Users/abinavrameshsundararaman/Documents/McGill/Courses/Winter 2019/Social Media Analytics/Final Project_Social_Media/clean_airpower.xlsx")

clean_twitter_data.describe()

## Calculate all centrality measures

G = nx.DiGraph()

#G=nx.from_pandas_edgelist(clean_twitter_data, source='column1',target= 'column2', edge_attr=True)
#G=nx.from_pandas_edgelist(clean_twitter_data, source='column1',target= 'column2')


lst=list()
for (a,b) in zip(clean_twitter_data.column1,clean_twitter_data.column2):
    lst.append((a,b))

G.add_edges_from(lst)

## import all centrality measures from gephi
centrality_measures = pd.read_csv("centrality_measures.csv")

# import users data : 
users_data = pd.read_excel("all_users_info_remove_dup.xlsx")
#users_data.distinct()
users_data=users_data.drop_duplicates()
t=users_data[users_data.screen_name=='Apple']
t=t.drop_duplicates()


users_data.groupby('screen_name')['screen_name'].count().sort_values(ascending=False)

# join users with their respective centrality measures
merged_users=pd.merge(users_data,centrality_measures,how="inner",left_on = 'screen_name', right_on='Id')

merged_users=merged_users.fillna(0)

###################### Calculate influence

## Standardize all columns

merged_users.columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


temp=merged_users.loc[:,['closnesscentrality','betweenesscentrality','outdegree','followers_count','listed_count']]
temp_scaled=scaler.fit_transform(temp)
merged_users_scaled=pd.DataFrame(temp_scaled)
merged_users_scaled.columns = ['closnesscentrality','betweenesscentrality','outdegree','followers_count','listed_count']

merged_users_scaled['screen_name'] = merged_users.screen_name

## Calculate influence

merged_users_scaled['Influence'] = 0.2*merged_users_scaled['closnesscentrality']+0.2*merged_users_scaled['betweenesscentrality']+0.2*merged_users_scaled['outdegree']+0.2*merged_users_scaled['followers_count']+0.2*merged_users_scaled['listed_count']

merged_users_scaled=merged_users_scaled.sort_values('Influence',ascending=False)
temp=merged_users_scaled.head(200)
temp.to_csv("top_200_influencers.csv")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Sentiment Analysis of all tweets

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
airpower.columns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

# function to print sentiments 
# of the sentence. 
def sentiment_scores(sentence): 
	sid_obj = SentimentIntensityAnalyzer() 
	sentiment_dict = sid_obj.polarity_scores(sentence) 
	return sentiment_dict


neg=list()
neu=list()
pos=list()
compound=list()

text=list(airpower.text)
for i in text:
    senti_dict=sentiment_scores(i)
    neg.append(senti_dict["neg"])
    neu.append(senti_dict["neu"])
    pos.append(senti_dict["pos"])
    compound.append(senti_dict["compound"])

airpower['neg'] = neg
airpower['pos'] = pos
airpower['neu'] = neu
airpower['compound'] = compound



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Calculate average sentiment per user

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
airpower.columns

avg_user_sentiment=airpower.groupby(['user_id','screen_name'])['compound'].mean().reset_index()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Did the sentiments of users change after the cancellation

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


airpower['is_after']=airpower['created_at'] >='2019-03-29'

user_sentiment=airpower.groupby(['user_id','screen_name','is_after'])['compound'].mean().reset_index()

user_sentiment=user_sentiment.fillna(0)

table = pd.pivot_table(user_sentiment, values=['compound'], index=['user_id','screen_name'],columns=['is_after'])

user_sentiment_before_after=pd.DataFrame(table).reset_index()
user_sentiment_before_after.columns = ['user_id','screen_name','compound_before','compound_after']

user_sentiment_before_after['sentiment_changed'] = ((user_sentiment_before_after['compound_before']>0)*(user_sentiment_before_after['compound_after']<0)+(user_sentiment_before_after['compound_before']<0)*(user_sentiment_before_after['compound_after']>0))


temp=user_sentiment_before_after[user_sentiment_before_after.sentiment_changed==True]


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Get Locations for each user-- most negative users are from which city and country?

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from geotext import GeoText

#drop na. purpose: avoid erro

cities=[]
countries=[]

for i in merged_users['location'] : 
    places = GeoText(str(i))
    if len(places.cities)==0 : 
        cities.append("")
    else : 
        cities.append(places.cities[0])
    if len(places.countries)==0:
        countries.append('')
    else : 
        countries.append(places.countries[0])
        

merged_users['City'] = cities
merged_users['Country'] = countries

merged_users[merged_users.Country!=""]
merged_users[merged_users.City!=""]

## Create a table of location for users

pd.merge(user_sentiment_before_after,merged_users)

merged_sentiments_location=pd.merge(merged_users,user_sentiment_before_after,how="left",left_on = 'screen_name',right_on='screen_name')
merged_sentiments_location.columns

# Merge cities and countries : If city os not present, compliment it with country

merged_sentiments_location['City_Country'] = merged_sentiments_location['City']
city = list(merged_sentiments_location['City'])
country = list(merged_sentiments_location['Country'])
city_country = list(merged_sentiments_location['City'])
length=len(merged_sentiments_location)
for i in range(length):
    if city[i]=="":
        city_country[i] = country[i]

merged_sentiments_location['City_Country'] = city_country


""""""""""""""""""""""""""""
Negative users city_country

"""""""""""""""""""""""""""""
# Negative users before cancellation are from which country??
negative_users_before = merged_sentiments_location[merged_sentiments_location.compound_before< (-0.5)]

t_negative_before=negative_users_before[negative_users_before.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()

# Negative users AFTER cancellation are from which country??
negative_users_after = merged_sentiments_location[merged_sentiments_location.compound_after< (-0.5)]

t_negative_after=negative_users_after[negative_users_after.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()


""""""""""""""""""""""""""""
Positive users country

"""""""""""""""""""""""""""""

# positive users before cancellation are from which country??
positive_users_before = merged_sentiments_location[merged_sentiments_location.compound_before> (0.5)]

t_positive_before=positive_users_before[positive_users_before.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()

# positive users AFTER cancellation are from which country??
positive_users_after = merged_sentiments_location[merged_sentiments_location.compound_after> (0.5)]

t_positive_after=positive_users_after[positive_users_after.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Get Locations for each tweet-- most negative tweets are from which city and country?

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from geotext import GeoText

#drop na. purpose: avoid erro

cities=[]
countries=[]

for i in airpower['location'] : 
    places = GeoText(str(i))
    if len(places.cities)==0 : 
        cities.append("")
    else : 
        cities.append(places.cities[0])
    if len(places.countries)==0:
        countries.append('')
    else : 
        countries.append(places.countries[0])
        

airpower['City'] = cities
airpower['Country'] = countries

airpower[airpower.Country!=""]
airpower[airpower.City!=""]

## Create City_Country columns
airpower['City_Country'] = airpower['City']
city = list(airpower['City'])
country = list(airpower['Country'])
city_country = list(airpower['City'])
length=len(airpower)
for i in range(length):
    if city[i]=="":
        city_country[i] = country[i]

airpower['City_Country'] = city_country


before_cancel=airpower[(airpower['created_at'] <'2019-03-29') ]
after_cancel=airpower[(airpower['created_at'] >='2019-03-29') ]

""""""""""""""""""""""""""""
Location with the most negative tweets

"""""""""""""""""""""""""""""

list(before_cancel['compound'])
# Negative tweets before cancellation are from which location??
negative_tweets_before = before_cancel[before_cancel['compound']< (-0.5)]

t_negative_tweets_before=negative_tweets_before[negative_tweets_before.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()


# Negative tweets after cancellation are from which location??
negative_tweets_after = after_cancel[after_cancel['compound']< (-0.5)]

t_negative_tweets_after=negative_tweets_after[negative_tweets_after.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()


## Wordcloud
d = {}
for a, x in t_negative_tweets_after.values:
    d[a] = x

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

""""""""""""""""""""""""""""
Location with the most positive tweets

"""""""""""""""""""""""""""""

list(before_cancel['compound'])
# positive tweets before cancellation are from which location??
positive_tweets_before = before_cancel[before_cancel['compound']> (0.5)]

t_positive_tweets_before=positive_tweets_before[positive_tweets_before.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()


# positive tweets after cancellation are from which location??
positive_tweets_after = after_cancel[after_cancel['compound']>(0.5)]

t_positive_tweets_after=positive_tweets_after[positive_tweets_after.City_Country !=""].groupby('City_Country')['screen_name'].count().reset_index()


##Wprdcloud
d = {}
for a, x in t_positive_tweets_after.values:
    d[a] = x

import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Sentiment Analysis of the the top influencers : 
    1. Does top influencers are positive or negative?
    2. Where are the top influencers located?
    3. Network to highlight influencers

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


merged_users_scaled.columns = ['closnesscentrality_scaled', 'betweenesscentrality_scaled', 'outdegree_scaled','followers_count_scaled', 'listed_count_scaled', 'screen_name', 'Influence']
merged_sentiments_location.columns

merged_users=pd.merge(merged_sentiments_location,merged_users_scaled,how="left",on="screen_name").sort_values(by="Influence",ascending=False)

# To include "compound" score for all users
merged_users=pd.merge(merged_users,avg_user_sentiment,how="left",on="screen_name").sort_values(by="Influence",ascending=False)



#########   1. Does top influencers are positive or negative?

top_200_influencers=merged_users.iloc[:200,:]

temp=top_200_influencers.loc[:,['screen_name','compound','Influence']]
temp=temp.fillna(0)
temp = temp.assign(is_positive = ['Neutral' if (a==0) else ('Positive' if (a>0) else 'Negative') for a in temp['compound']])
temp.groupby('is_positive')['screen_name'].count()

# before announcement
temp=top_200_influencers.loc[:,['screen_name','compound_before','Influence']]
temp=temp.fillna(0)
temp = temp.assign(is_positive = ['Neutral' if (a==0) else ('Positive' if (a>0) else 'Negative') for a in temp['compound_before']])
temp.groupby('is_positive')['screen_name'].count()

# After announcement
temp=top_200_influencers.loc[:,['screen_name','compound_after','Influence']]
temp=temp.fillna(0)
temp = temp.assign(is_positive = ['Neutral' if (a==0) else ('Positive' if (a>0) else 'Negative') for a in temp['compound_after']])
temp.groupby('is_positive')['screen_name'].count()



############## 2. Where are the top influencers located?


temp=top_200_influencers.groupby("City_Country")["screen_name"].count().reset_index().sort_values(by="screen_name",ascending=False)


############## 3. Were the top influencers spreading positive or negative sentiments


temp=top_200_influencers.loc[:,['screen_name','City_Country','compound','Influence']]
temp=temp.fillna(0)
temp = temp.assign(is_positive = ['Neutral' if (a==0) else ('Positive' if (a>0) else 'Negative') for a in temp['compound']])
country_pos_neg=temp.groupby(['City_Country','is_positive'])['screen_name'].count().reset_index()

table_country_pos = pd.pivot_table(country_pos_neg, values=['screen_name'], index=['City_Country'],columns=['is_positive']).reset_index()
table_country_pos.columns = ['Country','Negative','Neutral','Positive']
table_country_pos=table_country_pos.fillna(0)

############## 4. Who are the top influencers to target in each location

top_200_influencers.columns
location="India"
def target(df,loc,pos):
    if pos : 
        return(df[(df['location']==str(loc))&(df['compound']>0) ])
    else : 
        return(df[(df['location']==str(loc))&(df['compound']<0) ])

target_loc = target(top_200_influencers,"New York",True)   
target_loc['screen_name'].head(10)
target_loc = target(merged_users,"San Francisco",True)   
target_loc['screen_name'].head(10)
target_loc = target(merged_users,"India",True)   
target_loc['screen_name'].head(10)
target_loc = target(merged_users,"United States",False)   
target_loc['screen_name'].head(10)


######################################################################

# Pick top 200 pos influencers before
top_200_pos_influencers_before=merged_users[merged_users.compound_before>0].iloc[:200,:]
top_200_neg_influencers_before=merged_users[merged_users.compound_before<0].iloc[:200,:]
top_200_pos_influencers_after=merged_users[merged_users.compound_after>0].iloc[:200,:]
top_200_neg_influencers_after=merged_users[merged_users.compound_after<0].iloc[:200,:]

# see which location they are from
top_200_pos_influencers_before.groupby("City_Country")["screen_name"].count().reset_index().sort_values(by="screen_name",ascending=False)

top_200_neg_influencers_before.groupby("City_Country")["screen_name"].count().reset_index().sort_values(by="screen_name",ascending=False)

top_200_pos_influencers_after.groupby("City_Country")["screen_name"].count().reset_index().sort_values(by="screen_name",ascending=False)

top_200_pos_influencers_after.groupby("City_Country")["screen_name"].count().reset_index().sort_values(by="screen_name",ascending=False)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
What were the sentiments before and after sentiments for EACH LOCATION? (Aggregate at the location level )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
airpower.columns
t1=airpower.groupby('City_Country')['compound'].mean().reset_index()
t1['ispositive'] = t1['compound']>0
t1['ispositive'].mean()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Topic modelling

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


airpower.columns


airpower_LDA_df=airpower.loc[:,["City_Country","text"]]

#checking for nulls if present any
print("Number of rows with any of the empty columns:")
print(airpower_LDA_df.isnull().sum().sum())
airpower_LDA_df=airpower_LDA_df.dropna() 

engagement_score = 'City_Country'
labels = 'text'
ntopics= 4

word_tokenizer=RegexpTokenizer(r'\w+')
wordnet_lemmatizer = WordNetLemmatizer()
stopwords_nltk=set(stopwords.words('english'))

def tokenize_text(version_desc):
    ps = nltk.stem.PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    lowercase=version_desc.lower()
    text = wordnet_lemmatizer.lemmatize(lowercase)
    tokens = word_tokenizer.tokenize(text)
    token1=[]
    for i in tokens : 
        if i.isalpha():
            token1.append(i)
    return token1

vec_words = CountVectorizer(tokenizer=tokenize_text,stop_words=stopwords_nltk,decode_error='ignore')
total_features_words = vec_words.fit_transform(airpower_LDA_df[labels])

print(total_features_words.shape)

model = lda.LDA(n_topics=int(ntopics), n_iter=500, random_state=1)
model.fit(total_features_words)

topic_word = model.topic_word_ 
doc_topic=model.doc_topic_
doc_topic=pd.DataFrame(doc_topic)
airpower_LDA_df=airpower_LDA_df.join(doc_topic)
engagement=pd.DataFrame()

for i in range(int(ntopics)):
    topic="topic_"+str(i)
    engagement[topic]=airpower_LDA_df.groupby([engagement_score])[i].mean()
    
engagement=engagement.reset_index()
topics=pd.DataFrame(topic_word)
topics.columns=vec_words.get_feature_names()
topics_df=topics.transpose()
topics_df.to_excel("topic_word_dist.xlsx")
engagement.to_excel("engagement_topic_dist.xlsx",index=False)
