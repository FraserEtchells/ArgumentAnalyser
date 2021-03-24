#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import string
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from nltk.tag.stanford import StanfordPOSTagger
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

nltk.download('averaged_perceptron_tagger')

list_of_pos_tags = [',','.',':','``',"''",'CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

#Run this feature first so we have each sentences lemmatized and their N-grams labelled
def tokenisation_features(data):
    nlp = spacy.load("en_core_web_md")
    
    lemmatized_sentences_tokens = []
    lemmatized_sentences_joined = []
    bigrams = []
    for index, row in data.iterrows():
        sentence = row['Sentence']
        tokenized = nlp(sentence)
        sentence_lemmatized = []
        for word in tokenized:
            sentence_lemmatized.append(word.lemma_)
        
        lemmatized_sentences_tokens.append(sentence_lemmatized)
        lemmatized_sentences_joined.append(' '.join(sentence_lemmatized))
        sentence_bigrams = list(nltk.bigrams(sentence_lemmatized))
        bigrams.append(sentence_bigrams)
       
    data['Lemmatized Sentence'] = lemmatized_sentences_joined
    data['Lemmatized Sentence Tokens'] = lemmatized_sentences_tokens
    data['Lemmatized Sentence Bigrams'] = bigrams

def part_of_speech_features(data):
    
    for tag in list_of_pos_tags:
        string_tag = "Distribution of " + tag + " POS Tag"
        data[string_tag] = 0.0
        
    part_of_speech_tokens = []
    most_common_pos_token = []
    
    for index, row in data.iterrows():
        dict_of_occurences = {}
        sentence_tokens = row['Lemmatized Sentence Tokens']
        
        pos_tokens = nltk.pos_tag(sentence_tokens)
        
        tokens, pos_tags = zip(*pos_tokens)
        
        part_of_speech_tokens.append(pos_tags)
        
        for curr_tag in pos_tags:
            curr_tag_occurences = pos_tags.count(curr_tag)
            #simple bug fix - these symbols keep occuring randomly
            if curr_tag == "(":
                continue
            if curr_tag == ")":
                continue
            if curr_tag == "$":
                continue
            if curr_tag in dict_of_occurences:
                continue
            else:
                dict_of_occurences[curr_tag] = curr_tag_occurences
                string_curr_tag = "Distribution of " + curr_tag + " POS Tag"
                data.loc[index, string_curr_tag] = curr_tag_occurences / len(pos_tags)
            
def positional_features(data):
    
    within_introduction = []
    within_conclusion = []
    essays_in_dataframe = set()
    
    #When iterating through all of the data, also append the essay id to a list, with only unique values, so we know the essays being used in this block.
    for index, row in data.iterrows():
        paragraph = row['Paragraph Number']
        essays_in_dataframe.add(row['Essay ID'])
        
        if paragraph == 2:
            within_introduction.append(1)
            within_conclusion.append(0)
            
        elif paragraph == row['Total Paragraphs']:
            within_introduction.append(0)
            within_conclusion.append(1)
            
        else:
            within_introduction.append(0)
            within_conclusion.append(0)
    data['Sentence Within Introduction'] = within_introduction
    data['Sentence Within Conclusion'] = within_conclusion
    
    number_of_components_proceeding = []
    number_of_components_preceeding = []
    
    #Go through each essay and each paragraph. Total number of components in the paragraph = all sentences in the paragraph
    #Want to extract the order of components
    for id,curr_essay_id in enumerate(essays_in_dataframe): 
        curr_essay = data.loc[(data['Essay ID'] == curr_essay_id)]
        curr_essay_total_paragraphs = curr_essay['Total Paragraphs'].values[0]
        for curr_paragraph_number in range(curr_essay_total_paragraphs):
            curr_paragraph = curr_essay.loc[(curr_essay['Paragraph Number'] == curr_paragraph_number + 1)]
            total_components_in_paragraph = len(curr_paragraph.index)
            for curr_sentence_number in range(total_components_in_paragraph):
                number_of_components_proceeding.append(total_components_in_paragraph - (curr_sentence_number+1))
                number_of_components_preceeding.append(curr_sentence_number)
                
    data['Number of Proceeding Components'] = number_of_components_proceeding
    data['Number of Preceding Components'] = number_of_components_preceeding
            
def first_person_indicators_features(data):
    
    first_person_indicators = set(['i', 'myself', 'my', 'mine'])
    presence_of_indicators = []
    count_of_indicators = []
    #Load in each sentence. Go through each word. If the current word is within the list, set presence to true and count + 1
    
    for index, row in data.iterrows():
        sentence = row['Sentence']
        sentence_tokens = nltk.word_tokenize(sentence)
        current_count = 0
        indicator_present = False
        
        for word in sentence_tokens:
            lowercase_word = word.lower()
            if lowercase_word in first_person_indicators:
                indicator_present = True
                current_count += 1
                
        if indicator_present == True:
            presence_of_indicators.append(1)
        else:
            presence_of_indicators.append(0)
            
        count_of_indicators.append(current_count)
    data['First Person Indicator Present'] = presence_of_indicators
    data['First Person Indicator Count'] = count_of_indicators
    

def forward_indicator_feature(data):
    #therefore/thus/consequently indicate the component after the indicator may be a claim
    #take each word, make a copy lowercase, check if it is in list
    forward_indicators = ['therefore' , 'thus', 'consequently']
    presence_of_indicators = []
    for index, row in data.iterrows():
        sentence = row['Sentence']
        sentence_tokens = nltk.word_tokenize(sentence)
        current_count = 0
        indicator_present = False
        
        for word in sentence_tokens:
            lowercase_word = word.lower()
            if lowercase_word in forward_indicators:
                indicator_present = True
                
        if indicator_present == True:
            presence_of_indicators.append(1)
        else:
            presence_of_indicators.append(0)
        
    data['Forward Indicator Present'] = presence_of_indicators
    
def backward_indicator_feature(data):
    #in addition/because/additionally indicate the component after the indicator may be a premise
   
    backward_indicators = ['in addition', 'because', 'additionally']
    presence_of_indicators = []
    
    for index, row in data.iterrows():
        lower_case_sentence = row['Sentence'].lower()
        indicator_present = False
        
        for i in range(len(backward_indicators)):
            if (lower_case_sentence.find(backward_indicators[i]) != -1):
                indicator_present = True
        
        if indicator_present == True:
            presence_of_indicators.append(1)
        else:
            presence_of_indicators.append(0)
            
    data['Backward Indicator Present'] = presence_of_indicators
        
    
def thesis_indicator_feature(data):
    #in my opinion/I believe indicate a component after the indicator may be a major claim

    thesis_indicators = ['in my opinion','in my honest opinion' ,'i believe', 'i firmly believe', 'i strongly believe']
    
    presence_of_indicators = []
    
    for index, row in data.iterrows():
        lower_case_sentence = row['Sentence'].lower()
        indicator_present = False
        
        for i in range(len(thesis_indicators)):
            if (lower_case_sentence.find(thesis_indicators[i]) != -1):
                indicator_present = True
        
        if indicator_present == True:
            presence_of_indicators.append(1)
        else:
            presence_of_indicators.append(0)
            
    data['Thesis Indicator Present'] = presence_of_indicators
        
                
train = pd.read_pickle("./train.pkl")
test = pd.read_pickle("./test.pkl")

test_essay_id = 4
test_essay = test.loc[(test['Essay ID'] == test_essay_id)]

feature_columns=['Lemmatized Sentence','Paragraph Number', 'Sentence Within Introduction', 'Sentence Within Conclusion', 'Number of Proceeding Components', 'Number of Preceding Components' , 'First Person Indicator Present', 'First Person Indicator Count', 'Forward Indicator Present', 'Backward Indicator Present', 'Thesis Indicator Present']
for curr_tag in list_of_pos_tags:
    feature_columns.append("Distribution of " + curr_tag + " POS Tag")

#Remove all non-argumentative sentences from the train and test pool. This is to simulate the Identification process identifying the argumentative sentences.

non_argumentative_train = train[ train['Argumentative Label'] == '0'].index
train.drop(non_argumentative_train, inplace = True)
train.reset_index(drop=True, inplace=True)

non_argumentative_test = test[ test['Argumentative Label'] == '0'].index
test.drop(non_argumentative_test, inplace = True)
test.reset_index(drop=True, inplace=True)

tokenisation_features(train)
part_of_speech_features(train)
positional_features(train)
first_person_indicators_features(train)
forward_indicator_feature(train)
backward_indicator_feature(train)
thesis_indicator_feature(train)

tokenisation_features(test)
part_of_speech_features(test)
positional_features(test)
first_person_indicators_features(test)
forward_indicator_feature(test)
backward_indicator_feature(test)
thesis_indicator_feature(test)

tokenisation_features(test_essay)
part_of_speech_features(test_essay)
positional_features(test_essay)
first_person_indicators_features(test_essay)
forward_indicator_feature(test_essay)
backward_indicator_feature(test_essay)
thesis_indicator_feature(test_essay)
print(test_essay)



#Y should be the argument component type label encoded - labels being MajorClaim, Claim and Premise
component_type = preprocessing.LabelEncoder()
#for some reason, MajorClaim = 1 while Claim = 0. Unsure why this is but keep in mind for testing label encoding
component_type.fit(['MajorClaim','Claim', 'Premise'])

x = train.loc[:, feature_columns]
print(x)
y = train.loc[:, ['Argument Component Type']]
y_encoded = component_type.transform(y)
y['Argument Component Type'] = y_encoded

x_new = test.loc[:, feature_columns]
print(x_new)
y_new = test.loc[:, ['Argument Component Type']]
print(y_new)
y_new_encoded = component_type.transform(y_new)
print(y_new_encoded)
y_new['Argument Component Type'] = y_new_encoded

tf = TfidfVectorizer(max_features = 800,strip_accents = 'ascii',stop_words = 'english',)

x_sentences = x['Lemmatized Sentence']

x_sentences_vectorized = tf.fit_transform(x_sentences)
x_vectorized_dataframe = pd.DataFrame(x_sentences_vectorized.todense(), columns=tf.get_feature_names())
x_concat = pd.concat([x, x_vectorized_dataframe], axis=1)
x_final = x_concat.drop(['Lemmatized Sentence'], axis=1)

x_new_sentences = x_new['Lemmatized Sentence']

x_new_sentences_vectorized = tf.transform(x_new_sentences)
x_new_vectorized_dataframe = pd.DataFrame(x_new_sentences_vectorized.todense(), columns=tf.get_feature_names())
x_new_concat = pd.concat([x_new, x_new_vectorized_dataframe], axis=1)
x_new_final = x_new_concat.drop(['Lemmatized Sentence'], axis=1)

naive_bayes = MultinomialNB()
naive_bayes.fit(x_final,y.values.ravel())

predictions = naive_bayes.predict(x_new_final)

baseline = predictions
baseline = np.where(baseline < 2, 2, baseline)

c_m = confusion_matrix(y_new.values.ravel(), predictions)
c_m_true = confusion_matrix(y_new.values.ravel(), y_new.values.ravel())
print('Predicted Values: ', predictions)
print('Accuracy score: ', accuracy_score(y_new.values.ravel(), predictions))
print('Precision score: ', precision_score(y_new.values.ravel(), predictions, average='weighted'))
print('Recall score: ', recall_score(y_new.values.ravel(), predictions, average='weighted'))
print('Baseline Accuracy score: ', accuracy_score(y_new.values.ravel(), baseline))
print('Baseline Precision score: ', precision_score(y_new.values.ravel(), baseline, average='weighted'))
print('Baseline Recall score: ', recall_score(y_new.values.ravel(), baseline, average='weighted'))
print('Confusion Matrix:')
print(c_m)
print('Actual Result Matrix:')
print(c_m_true)


# In[ ]:




