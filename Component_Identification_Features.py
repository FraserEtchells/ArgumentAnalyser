#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import json
import random
import sys
import ast
import numpy as np
import argparse
import nltk
import pickle
import scipy
import spacy
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

nltk.download('averaged_perceptron_tagger')

pos_tags = [',','.',':','``',"''",'CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
   
def position_features(data):
    dataframe = data
    
    start_positions = []
    end_positions = []
    for index, row in dataframe.iterrows():
        paragraph = row['Source Paragraph']
        sentence = row['Sentence']
        start_pos = paragraph.find(sentence)
        end_pos=sentence.find(sentence[-1:]) + start_pos
        
        start_positions.append(start_pos)
        end_positions.append(end_pos)
        
    dataframe['Relative Sentence Start Pos'] = start_positions
    dataframe['Relative Sentence End Pos'] = end_positions
        
               
def token_features(data):
    dataframe = data
    
    part_of_speech_tokens = []
    most_common_pos_token = []
    
    for index, row in dataframe.iterrows():
        sentence = row['Sentence']
        
        sentence_tokens = nltk.word_tokenize(sentence)
        pos_tokens = nltk.pos_tag(sentence_tokens)
        
        tokens, pos_tags = zip(*pos_tokens)
        
        part_of_speech_tokens.append(pos_tags)
        most_common_pos_token.append(max(set(pos_tags), key=pos_tags.count))
        
    dataframe['Sentence POS Tokens'] = part_of_speech_tokens
    dataframe['Most Common POS Token'] = most_common_pos_token


def similarity_features(data):
    dataframe = data
    nlp = spacy.load("en_core_web_md")
    
    similarities = []
    
    for index, row in dataframe.iterrows():
        essay_id = row['Essay ID']
        prompt_dataframe = dataframe.loc[(dataframe['Essay ID'] == essay_id)& (dataframe['Paragraph Number'] == 1)]
        prompt = prompt_dataframe.iloc[0]['Sentence']
        sentence = row['Sentence']
        prompt_doc = nlp(prompt.lower())
        sentence_doc = nlp(sentence.lower())
        
        prompt_result = []
        sentence_result = []
        
        #Following code was obtained from a tutorial - https://medium.com/better-programming/the-beginners-guide-to-similarity-matching-using-spacy-782fc2922f7c
        for token in prompt_doc:
            if token.text in nlp.Defaults.stop_words: 
                continue
            if token.is_punct:
                continue
            prompt_result.append(token.text)
            
        for token in sentence_doc:
            if token.text in nlp.Defaults.stop_words: 
                continue
            if token.is_punct:
                continue
            sentence_result.append(token.text)
        
        new_prompt = nlp(" ".join(prompt_result))
        new_sentence = nlp(" ".join(sentence_result))
        
       
        similarities.append(new_prompt.similarity(new_sentence))
    dataframe['Sentence Similarity To Prompt'] = similarities

def main():
    train = pd.read_pickle("./train.pkl")
    test = pd.read_pickle("./test.pkl")

    position_features(train)
    token_features(train)
    similarity_features(train)

    position_features(test)
    token_features(test)
    similarity_features(test)

    feature_columns=['Sentence', 'Sentence Similarity To Prompt','Most Common POS Token' ]

    tf = TfidfVectorizer(max_features = 800,strip_accents = 'ascii',stop_words = 'english',)
    le = preprocessing.LabelEncoder()
    pos_encoder = preprocessing.LabelEncoder()
    pos_encoder.fit(pos_tags)

    x = train.loc[:, feature_columns]
    y = train.loc[:, ['Argumentative Label']]
    x_sentences = x['Sentence']

    x_sentences_vectorized = tf.fit_transform(x_sentences)
    x_vectorized_dataframe = pd.DataFrame(x_sentences_vectorized.todense(), columns=tf.get_feature_names())
    x_concat = pd.concat([x, x_vectorized_dataframe], axis=1)
    x_final = x_concat.drop(['Sentence'], axis=1)

    x_pos_encoded = pos_encoder.transform(x['Most Common POS Token'])
    x_final['Most Common POS Token'] = x_pos_encoded

    y_binarized = le.fit_transform(y)
    y['Argumentative Label'] = y_binarized

    x_new = test.loc[:, feature_columns]
    y_new = test.loc[:, ['Argumentative Label']]
    x_new_sentences = x_new['Sentence']

    x_new_sentences_vectorized = tf.transform(x_new_sentences)
    x_new_vectorized_dataframe = pd.DataFrame(x_new_sentences_vectorized.todense(), columns=tf.get_feature_names())
    x_new_concat = pd.concat([x_new, x_new_vectorized_dataframe], axis=1)
    x_new_final = x_new_concat.drop(['Sentence'], axis=1)

    x_new_pos_encoded = pos_encoder.transform(x_new['Most Common POS Token'])
    x_new_final['Most Common POS Token'] = x_new_pos_encoded

    y_new_binarized = le.transform(y_new)
    y_new['Argumentative Label'] = y_new_binarized

    naive_bayes = MultinomialNB()
    naive_bayes.fit(x_final,y.values.ravel())

    predictions = naive_bayes.predict(x_new_final)

    test['Predicted Argumentative Label'] = predictions

    #pickle.dump(tf, open("tfidf.pickle", "wb"))
    #pickle.dump(pos_encoder, open("pos_encoder.pickle", "wb"))
    #pickle.dump(le, open("arg_label_encoder.pickle", "wb"))
    #pickle.dump(naive_bayes, open("component_identification_model.pickle", "wb"))

    baseline = predictions
    baseline = np.where(baseline < 1, 1, baseline)

    c_m = confusion_matrix(y_new.values.ravel(), predictions)

    print('Predicted Values: ', predictions)
    print('Accuracy score: ', accuracy_score(y_new.values.ravel(), predictions))
    print('Precision score: ', precision_score(y_new.values.ravel(), predictions, average='weighted'))
    print('Recall score: ', recall_score(y_new.values.ravel(), predictions, average='weighted'))
    print('Baseline Accuracy score: ', accuracy_score(y_new.values.ravel(), baseline))
    print('Baseline Precision score: ', precision_score(y_new.values.ravel(), baseline, average='weighted'))
    print('Baseline Recall score: ', recall_score(y_new.values.ravel(), baseline, average='weighted'))
    print('Confusion Matrix:')
    print(c_m)


# In[ ]:




