import time

import matplotlib.pyplot as plt
import bs4
import tweepy,textblob
import re
import pandas as pd
import numpy as np
import nltk
import re
# nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.patches as mpaches
from tensorflow.keras.callbacks import TensorBoard

NAME = "SENTIMENTal_TEXT_CLASSIFIER-3CAT-5000X50-{}".format(int(time.time()))
#creating a log dir
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

#loading our data
df1 = pd.read_csv("Twitter_Data.csv")
df2 = pd.read_csv("Reddit_Data.csv")

df1 = df1.rename(columns={'clean_text':'text','category':'sentiment'})
df1['sentiment'] = df1['sentiment'].map({-1.0:-1, 0.0:0, 1.0:1})
# print(df1.head())
df2 = df2.rename(columns={"clean_comment":"text","category":"sentiment"})
# print(df2.head())

#uncleaned datasets
df3 = pd.read_csv("Tweets.csv")
df3 = df3[['text','airline_sentiment']]
df3 = df3.rename(columns={'text':'text','airline_sentiment':'sentiment'})
df3['sentiment'] = df3['sentiment'].map({'positive':1.0,'negative':-1.0,'neutral':0.0})
# print(df3.head())

df4 = pd.read_csv("Covid_Tweets.csv")
df4 = df4[['OriginalTweet','Sentiment']]
df4 = df4.rename(columns={'OriginalTweet':'text','Sentiment':'sentiment'})
df4['sentiment'] = df4['sentiment'].map({'Positive':1.0,'Extremely Positive':1.0,'Negative':-1.0,'Extremely Negative':-1.0,'Neutral':0.0})
# print(df4.head())

df5 = pd.read_csv("Corona_Tweets.csv")
df5 = df5[['OriginalTweet','Sentiment']]
df5 = df5.rename(columns={'OriginalTweet':'text','Sentiment':'sentiment'})
df5['sentiment'] = df5['sentiment'].map({'Positive':1.0,'Extremely Positive':1.0,'Negative':-1.0,'Extremely Negative':-1.0,'Neutral':0.0})
# print(df5.head())

def clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#[A-Za-z0-9]+','',text)
    text = re.sub(r'\\n','',text)
    text = re.sub(r'http?:\/\/\S+','',text)
    text = re.sub(r'https:\/\/\S+','',text)
    text = re.sub(r'RT[\s]+','',text)
    text = re.sub('_','',text)
    text = re.sub("b[(')]",'',text)
    text = re.sub('b[(")]','',text)
    text = re.sub("\'",'',text)
    text = re.sub(':','',text)

    return text
#cleaning the tweets and also preparing the data
df3['text'] = df3['text'].apply(clean_text)
# print(df3.head())
df4['text'] = df4['text'].apply(clean_text)
# print(df4.head())
df5['text'] = df5['text'].apply(clean_text)
# print(df5.head())

new_df = pd.concat([df1,df2,df3,df4],ignore_index=True)
# print(new_df.head())
new_df.dropna(axis=0,inplace=True)
# print(new_df.isnull().sum())
new_df['sentiment'] = new_df['sentiment'].map({-1.0:'Negative',1.0:'Positive',0.0:'Neutral'})
# print(new_df.head())

#data processing
def tweet_to_words(tweet):
    '''Convert tweet to a sequence of words'''
    text = tweet.lower()
    #remove non letters
    text = re.sub(r"[^a-zA-Z0-9]","",text)
    words = text.split()
    #remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    #apply stemming
    words = [PorterStemmer().stem(w) for w in words]

    return words
# print("Original Tweet : ",new_df['text'][0])
# print("Processed Tweet : ",tweet_to_words(new_df['text'][0]))

X = list(map(tweet_to_words,new_df['text']))

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(new_df['sentiment'])
# print(X[0])
# print(Y[0])

y = pd.get_dummies(new_df['sentiment'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=1)

from sklearn.feature_extraction.text import CountVectorizer
vocabulary_size = 5000

count_vector = CountVectorizer(max_features=vocabulary_size,
                               preprocessor=lambda x: x,
                               tokenizer=lambda x: x
                               )

# import sklearn.preprocessing as pr
# X_train = pr.normalize(X_train, axis=1)
# X_test  = pr.normalize(X_test, axis=1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_words = 5000
max_len = 50

def tokenize_pad_sequences(text):
    '''This function tokenize the input text into sequences of intergers and then
    pad each sequence to the same length'''
    #text tokenization
    tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer.fit_on_texts(text)
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    # return sequences
    return X, tokenizer
# print('Before Tokenization & Padding \n', new_df['text'][0])
X, tokenizer = tokenize_pad_sequences(new_df['text'])
# print('After Tokenization & Padding \n', X[0])

import pickle

# saving
print("writing tokenizer to disk...")
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("finished writing tokenizer to disk...")

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

y = pd.get_dummies(new_df['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print('Train Set ->', X_train.shape, y_train.shape)
print('Validation Set ->', X_val.shape, y_val.shape)
print('Test Set ->', X_test.shape, y_test.shape)

#coding our model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import datasets

from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import History

from tensorflow.keras import losses

vocab_size = 5000
embedding_size = 32
epochs=25
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
# Build model
model= Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

model.summary()
model.compile(
    optimizer=sgd,
    metrics=['accuracy'],
    loss="categorical_crossentropy"
)
batch_size = 64
print("Model initializing training...")
model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[tensorboard])

loss,accuracy = model.evaluate(X_test,y_test,verbose=0)
print("Loss: ",loss)
print("Model Accuracy: ",accuracy)
# print("")
# print("Saving model to disk...")
# model.save("SENTIMENTAl_TEXT_CLASSIFIER-3CAT-5000X50.h5")
# print("Finished saving...")






