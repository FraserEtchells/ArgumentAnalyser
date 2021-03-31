#!/usr/bin/env python
# coding: utf-8

# In[33]:


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

def get_component_ratios(component_tuple, includes_para_number):
    major_claims_to_claims = 0
    premises_to_claims = 0
    
    if includes_para_number == False:
        if(component_tuple[0] > 0.1):
            major_claims_to_claims = component_tuple[1]/component_tuple[0]
        else:
            major_claims_to_claims = component_tuple[1]/1
        
        if (component_tuple[1] > 0.1):
            premises_to_claims = component_tuple[2]/component_tuple[1]
        else:
            premises_to_claims = component_tuple[2]/1
            
    elif includes_para_number == True:
        if(component_tuple[1] > 0.1):
            major_claims_to_claims = component_tuple[2]/component_tuple[1]
        else:
            major_claims_to_claims = component_tuple[2]
        
        if (component_tuple[2] > 0.1):
            premises_to_claims = component_tuple[3]/component_tuple[2]
        else:
            premises_to_claims = component_tuple[3]
    
    return major_claims_to_claims,premises_to_claims 

#returns a tuple that contains the passed essays major claim, claim and premise counts.
def component_count_total(essay):
    major_claim_count = 0
    claim_count = 0
    premise_count = 0
    
    
    for index, row in essay.iterrows():
        if row["Argument Component Type"] == "MajorClaim":
            major_claim_count +=1
        elif row["Argument Component Type"] == "Claim":
            claim_count +=1
        elif row["Argument Component Type"] == "Premise":
            premise_count +=1
                
    
    return major_claim_count, claim_count, premise_count
                
    
#returns a tuple that contains the passed dataframe of essays average number of major claims, claims and premises.
def average_component_count(data):
    component_counts = []
    major_claims = 0
    claims = 0
    premises = 0
    completed_essay_id = set()
    
    for index,row in data.iterrows():
        curr_essay_id = row["Essay ID"]
        if curr_essay_id in completed_essay_id:
            continue
        else:
            completed_essay_id.add(curr_essay_id)
        curr_essay = data.loc[(data['Essay ID'] == curr_essay_id)]
        component_counts.append(component_count_total(curr_essay))
    
    for component_tuple in component_counts:
        major_claims += component_tuple[0]
        claims += component_tuple[1]
        premises += component_tuple[2]
    
    average_major_claims = major_claims / len(component_counts)
    average_claims = claims / len(component_counts)
    average_premises = premises / len(component_counts)
    
    
    return average_major_claims, average_claims, average_premises

#returns a list of tuples. Tuples take the form of: paragraph number,major claim count, claim count, premise count.
def component_count_paragraphs(essay):
    current_paragraph = 0
    total_paragraphs = essay["Total Paragraphs"].values[0]
    paragraph_components_tuples = []
    
    for current_paragraph in range(total_paragraphs):
        p_major_claim_count = 0
        p_claim_count = 0
        p_premise_count = 0
        curr_paragraph = essay.loc[(essay['Paragraph Number'] == current_paragraph + 1)]
        for index, row in curr_paragraph.iterrows():
            if row["Argument Component Type"] == "MajorClaim":
                p_major_claim_count +=1
            elif row["Argument Component Type"] == "Claim":
                p_claim_count +=1
            elif row["Argument Component Type"] == "Premise":
                p_premise_count +=1
        paragraph_tuple = (current_paragraph+1, p_major_claim_count, p_claim_count, p_premise_count)
        paragraph_components_tuples.append(paragraph_tuple)
    return paragraph_components_tuples

def average_introduction_component_count(data):
    introduction_component_counts = []
    major_claims = 0
    claims = 0
    premises = 0
    completed_essay_id = set()
    
    for index,row in data.iterrows():
        curr_essay_id = row["Essay ID"]
        if curr_essay_id in completed_essay_id:
            continue
        else:
            completed_essay_id.add(curr_essay_id)
        curr_essay = data.loc[(data['Essay ID'] == curr_essay_id)]
        paragraphs_list = component_count_paragraphs(curr_essay)
        introduction_component_counts.append(paragraphs_list[1])
    
    for component_tuple in introduction_component_counts:
        major_claims += component_tuple[1]
        claims += component_tuple[2]
        premises += component_tuple[3]
    
    average_major_claims = major_claims / len(introduction_component_counts)
    average_claims = claims / len(introduction_component_counts)
    average_premises = premises / len(introduction_component_counts)
    

    return average_major_claims, average_claims, average_premises
    
def average_conclusion_component_count(data):
    conclusion_component_counts = []
    major_claims = 0
    claims = 0
    premises = 0
    completed_essay_id = set()
    
    for index,row in data.iterrows():
        curr_essay_id = row["Essay ID"]
        if curr_essay_id in completed_essay_id:
            continue
        else:
            completed_essay_id.add(curr_essay_id)
        curr_essay = data.loc[(data['Essay ID'] == curr_essay_id)]
        paragraphs_list = component_count_paragraphs(curr_essay)
        conclusion_component_counts.append(paragraphs_list[-1])
    
    for component_tuple in conclusion_component_counts:
        major_claims += component_tuple[1]
        claims += component_tuple[2]
        premises += component_tuple[3]
    
    average_major_claims = major_claims / len(conclusion_component_counts)
    average_claims = claims / len(conclusion_component_counts)
    average_premises = premises / len(conclusion_component_counts)
    
    return average_major_claims, average_claims, average_premises
    
def average_paragraph_component_count(data):
    component_counts = []
    major_claims = 0
    claims = 0
    premises = 0
    completed_essay_id = set()
    
    for index,row in data.iterrows():
        curr_essay_id = row["Essay ID"]
        if curr_essay_id in completed_essay_id:
            continue
        else:
            completed_essay_id.add(curr_essay_id)
        curr_essay = data.loc[(data['Essay ID'] == curr_essay_id)]
        paragraphs_list = component_count_paragraphs(curr_essay)
        paragraphs_list.pop(0)
        paragraphs_list.pop(0)
        paragraphs_list.pop(len(paragraphs_list)-1)
        for i in range(len(paragraphs_list)):
            component_counts.append(paragraphs_list[i])
       
    for component_tuple in component_counts:
        major_claims += component_tuple[1]
        claims += component_tuple[2]
        premises += component_tuple[3]
    
    average_major_claims = major_claims / len(component_counts)
    average_claims = claims / len(component_counts)
    average_premises = premises / len(component_counts)
    
    return average_major_claims, average_claims, average_premises
    
#gives feedback based on how the passed essay compares to the corpus' average results. Do this in the form of ratios to ensure longer essays will be marked appropriately.
def component_count_feedback(essay):
    average_component_count_tuple = (1.59, 3.30, 8.57)
    essay_component_count_tuple = component_count_total(essay)
    
    essay_ratio_tuple = get_component_ratios(essay_component_count_tuple, False)
    major_claims_to_claims = essay_ratio_tuple[0]
    premises_to_claims = essay_ratio_tuple[1]
    
    average_ratio_tuple = get_component_ratios(average_component_count_tuple, False)
    average_major_claims_to_claims = average_ratio_tuple[0]
    average_premises_to_claims = average_ratio_tuple[1]
    
    feedback = []
    feedback.append("Your essay has " + str(essay_component_count_tuple[0]) +" Major Claims," + str(essay_component_count_tuple[1]) + " Claims and " + str(essay_component_count_tuple[2]) + " Premises.")
    feedback.append("Your essay has a ratio of " + str(major_claims_to_claims) + " of Claims to Major Claims.")#Want a higher ratio
    feedback.append("On average, essays we have seen have a ratio of " + str(average_major_claims_to_claims) )
    if major_claims_to_claims - average_major_claims_to_claims > -0.1 and major_claims_to_claims - average_major_claims_to_claims < 0.1:
        feedback.append("This is good - it means you have a good amount of sub-arguments to support your overall thesis.")
    elif major_claims_to_claims - average_major_claims_to_claims < -0.1:
        feedback.append("This is not great - you may have too few arguments to support your overall thesis.")
    elif major_claims_to_claims - average_major_claims_to_claims > 0.1:
        feedback.append("While you have a lot of Claims to Major Claims, be aware that having too many claims may unfocus your thesis statement.")
    feedback.append("Your essay has a ratio of " + str(premises_to_claims) + " of Premises to Claims.") #Want a higher ratio
    feedback.append("On average, essays we have seen have a ratio of"+ str(average_premises_to_claims))
    if premises_to_claims - average_premises_to_claims < 0.1:
        feedback.append("This is not great. Generally, we want more premises than claims in order to give better justification to our points.")
    elif premises_to_claims - average_premises_to_claims > -0.1:
        feedback.append("This is good - it means on average you have a lot of support for your points.")
        
    return feedback


def paragraph_component_count_feedback(essay):
    #originally, we used functions to derive these results (which are the same as the dataset is static) which vastly increases run times.
    average_introduction_component_tuple = (0.75, 0.22, 0.03)
    average_conclusion_component_tuple = (0.82, 0.61, 0.18)
    average_paragraph_component_tuple = (0.004, 0.96, 3.27)
    
    essay_paragraph_tuple_list = component_count_paragraphs(essay)
    essay_paragraph_tuple_list.pop(0) # remove prompt paragraph
    essay_introduction_component_tuple = essay_paragraph_tuple_list.pop(0) #get tuple for the introduction paragraph
    essay_conclusion_component_tuple = essay_paragraph_tuple_list.pop(len(essay_paragraph_tuple_list)-1) #get tuple for conclusion paragraph
    #as we use pop method, list contains only the main body paragraphs.
    
    introduction_ratio_tuple = get_component_ratios(essay_introduction_component_tuple, True)
    introduction_major_claims_to_claims = introduction_ratio_tuple[0]
    introduction_premises_to_claims = introduction_ratio_tuple[1]
    
    average_introduction_ratio_tuple = get_component_ratios(average_introduction_component_tuple, False)
    average_introduction_major_claims_to_claims = average_introduction_ratio_tuple[0]
    average_introduction_premises_to_claims = average_introduction_ratio_tuple[1]
    
    conclusion_ratio_tuple = get_component_ratios(essay_conclusion_component_tuple, True)
    conclusion_major_claims_to_claims = conclusion_ratio_tuple[0]
    conclusion_premises_to_claims = conclusion_ratio_tuple[1]
    
    average_conclusion_ratio_tuple = get_component_ratios(average_conclusion_component_tuple, False)
    average_conclusion_major_claims_to_claims = average_conclusion_ratio_tuple[0]
    average_conclusion_premises_to_claims = average_conclusion_ratio_tuple[1]
    
    average_paragraph_ratio_tuple = get_component_ratios(average_paragraph_component_tuple, False)
    average_paragraph_major_claims_to_claims = average_paragraph_ratio_tuple[0]
    average_paragraph_premises_to_claims = average_paragraph_ratio_tuple[1]
    
    feedback = []
    feedback.append("The introduction has " + str(essay_introduction_component_tuple[1]) + " Major Claims, " +  str(essay_introduction_component_tuple[2]) + " Claims and " + str(essay_introduction_component_tuple[3]) + "Premises")
    feedback.append("The introduction has a ratio of " + str(introduction_major_claims_to_claims) + " of Claims to Major Claims and a ratio of " + str(introduction_premises_to_claims) + " of Premises to Claims" )
    feedback.append("On average, essays we have seen have a ratio of " + str(average_introduction_major_claims_to_claims) + " of Claims to Major Claims and a ratio of " + str(average_introduction_premises_to_claims) + " of Premises to Claims")
    
    if(introduction_major_claims_to_claims - average_introduction_major_claims_to_claims > -1 and introduction_major_claims_to_claims - average_introduction_major_claims_to_claims < 1.1):
        feedback.append("Your Claims to Major Claims ratio is good - generally we want less Claims and more Major Claims in an introduction, but having the same number is fine.")
    elif(introduction_major_claims_to_claims - average_introduction_major_claims_to_claims > 1.1):
        feedback.append("Your Claims to Major Claims ratio is not good - having too many claims and not many Major Claims in your introduction makes your structure messier.")
        
    if(introduction_premises_to_claims - average_introduction_premises_to_claims < 1):
        feedback.append("Your Premises to Claims ratio is good - this means that you are keeping your introduction focussed on your thesis statement.")
    elif(introduction_premises_to_claims - average_introduction_premises_to_claims > 1):
        feedback.append("Your Premises to Claims ratio is not good - aim to have 1 or 0 premises in your introduction. The introduction should be primarily used to summarise your thesis.")
        
    for i in range(len(essay_paragraph_tuple_list)):
        feedback.append("Paragraph " + str(essay_paragraph_tuple_list[i][0]) + " has " + str(essay_paragraph_tuple_list[i][1]) + " Major Claims, " + str(essay_paragraph_tuple_list[i][2]) + " Claims and " + str(essay_paragraph_tuple_list[i][3]) + " Premises.")
        paragraph_ratio_tuple = get_component_ratios(essay_paragraph_tuple_list[i], True)
        paragraph_major_claims_to_claims = paragraph_ratio_tuple[0]
        paragraph_premises_to_claims = paragraph_ratio_tuple[1]
        feedback.append("This Paragraph has a ratio of " + str(paragraph_major_claims_to_claims) + " of Claims to Major Claims and a ratio of " + str(paragraph_premises_to_claims) + " of Premises to Claims")
        feedback.append("On average, essays we have seen have a ratio of " + str(average_paragraph_major_claims_to_claims) + " of Claims to Major Claims and a ratio of " + str(average_paragraph_premises_to_claims) + " of Premises to Claims")
        
        if(paragraph_major_claims_to_claims - average_paragraph_major_claims_to_claims > 0):
            feedback.append("Your Claims to Major Claims ratio is good - generally we want no Major Claims in any of our main body paragraphs.")
        elif(paragraph_major_claims_to_claims - average_paragraph_major_claims_to_claims < 0):
            feedback.append("Your Claims to Major Claims ratio is poor - this may be due to an error in our system, or you have a Major Claim in a main body paragraph. Try to remove it, or re-word the claim you are presenting.")
    
        if(paragraph_premises_to_claims - average_paragraph_premises_to_claims > -1 and paragraph_premises_to_claims - average_paragraph_premises_to_claims < 2):
            feedback.append("Your Premises to Claims ratio is great - each Claim needs roughly 3 to 4 Premises to properly back it up.")
        elif(paragraph_premises_to_claims - average_paragraph_premises_to_claims < -1):
            feedback.append("Your Premises to Claims ratio is poor - you should aim to add more Premises to this Claim in order to give it proper justification.")
        elif(paragraph_premises_to_claims - average_paragraph_premises_to_claims > 2):
            feedback.append("Your Premises to Claims ratio is higher than average - if you have a limited word count you may be better removing some of your premises in this paragraph and either create a new paragraph to support the overall thesis, or add more Premises to another paragraph.")
        
    feedback.append("The conclusion has " + str(essay_conclusion_component_tuple[1]) + " Major Claims, " + str(essay_conclusion_component_tuple[2]) +  " Claims and " + str(essay_conclusion_component_tuple[3]) + " Premises")
    feedback.append("The conclusion has a ratio of " + str(conclusion_major_claims_to_claims) + " of Claims to Major Claims and a ratio of " + str(conclusion_premises_to_claims) + " of Premises to Claims" )
    feedback.append("On average, essays we have seen have a ratio of " + str(average_conclusion_major_claims_to_claims) + " of Claims to Major Claims and a ratio of " + str(average_conclusion_premises_to_claims) + " of Premises to Claims")
    
    if(conclusion_major_claims_to_claims - average_conclusion_major_claims_to_claims > -1 and conclusion_major_claims_to_claims - average_conclusion_major_claims_to_claims < 2):
        feedback.append("Your Claims to Major Claims ratio is good - generally we want less Claims and more Major Claims in a conclusion, although having a single Claim in your conclusion or summarising your Claims is also a good idea.")
    elif(conclusion_major_claims_to_claims - average_conclusion_major_claims_to_claims > 2):
        feedback.append("Your Claims to Major Claims ratio is not good - having too many Claims in your conclusion impacts the readability of your final thesis statement.")
        
    if(conclusion_premises_to_claims - average_conclusion_premises_to_claims < 1):
        feedback.append("Your Premises to Claims ratio is good - this means that you are keeping your conclusion focussed on your thesis statement.")
    elif(conclusion_premises_to_claims - average_conclusion_premises_to_claims > 1):
        feedback.append("Your Premises to Claims ratio is not good - aim to have 1 or 0 premises in your conclusion. The conclusion should be primarily used to sum up your overall thesis of the essay.")
    
    return feedback
    
def paragraph_component_sequence(essay):
    #Pass in an essay. Take each paragraph. Store the order of components in a series - MC = 1, C = 2 and P = 3. So in a paragraph with Claim, Premise Premise, it would go (1,2,2)
    total_paragraphs = essay["Total Paragraphs"].values[0]
    essay_paragraphs_flow = []
    
    for current_paragraph in range(2, total_paragraphs + 1): #start at 2, as we want to ignore the essay prompt sentence.
        paragraph = essay.loc[(essay['Paragraph Number'] == current_paragraph)]
        paragraph_flow = []
        paragraph_flow.append(current_paragraph)
        for index, row in paragraph.iterrows():
            if row['Argument Component Type'] == "MajorClaim":
                paragraph_flow.append("MajorClaim")
            elif row['Argument Component Type'] == "Claim":
                paragraph_flow.append("Claim")
            elif row['Argument Component Type'] == "Premise":
                paragraph_flow.append("Premise")
            else:
                paragraph_flow.append("None")
        essay_paragraphs_flow.append(paragraph_flow)
        
    return essay_paragraphs_flow

def paragraph_flow_feedback(essay):
    essay_paragraphs_flow = paragraph_component_sequence(essay)
    introduction = essay_paragraphs_flow.pop(0)
    conclusion = essay_paragraphs_flow.pop(len(essay_paragraphs_flow) -1)
    
    feedback = []
    if not introduction:
        feedback.append("It appears your Introduction consists of one or fewer sentences, therefore we cannot comment on the structure of it.")
    else:
        feedback.append("The flow of Argument Components in your Introduction goes: " + str(introduction) +  " where 'None' labels a non-argumentative sentence.")
        #In an introduction, we want the Major Claims to either be in the first or last sentence.
        if introduction[1] == "MajorClaim":
            feedback.append("Your introduction starts with a Major Claim. This is great, it means you are immediately informing your reader of your main focus for the essay.")
        elif introduction[len(introduction) -1] == "Major Claim":
            feedback.append("Your introduction ends with a Major Claim. This is great, as you mention your thesis statement right before getting into your arguments.")
    
        for index in range(len(introduction) -1):
            if index != 1 or index!= len(introduction)-1:
                if introduction[index] == "MajorClaim":
                    feedback.append("Your introduction contains a Major Claim, however it is not in a great position. Try to keep your Major Claims to the first or last sentence for better readability.")
    
    for index in range(len(essay_paragraphs_flow)):
        paragraph = essay_paragraphs_flow[index]
        feedback.append("The flow of Argument Components in Paragraph" + str(paragraph[0]) + " goes: " + str(paragraph) + " where 'None' labels a non-argumentative sentence.")
        if paragraph[1] == "Claim":
            feedback.append("Paragraph "+ str(paragraph[0]) + " starts with a Claim. This is good, you immediately bring this sub-arguments main point forward.")
        elif paragraph[len(paragraph) -1] == "Claim":
            feedback.append("Paragraph "+ str(paragraph[0]) + " ends with a Claim. This is good, you are ending the paragraph with the point of your previous statements.")
        for index_two in range(len(paragraph) -1):
            if index_two != 1 and index_two != len(paragraph)-1:
                if paragraph[index] == "Claim":
                    feedback.append("This paragraph contains a Claim that is neither at the start or the end of the paragraph. This makes it harder for the reader to determine the actual point of this argument")
    
    if not conclusion:
        feedback.append("It appears your Conclusion consists of one or fewer sentences, therefore we cannot comment on the structure of it.")
    else:
        feedback.append("The flow of Argument Components in your Conclusion goes:"+ str(conclusion) + " where 'None' labels a non-argumentative sentence.")
        if conclusion[1] == "MajorClaim":
            feedback.append("Your conclusion starts with a Major Claim. This is great, it means you are immediately describing the thesis of your essay.")
        elif conclusion[len(conclusion) -1] == "Major Claim":
            feedback.append("Your conclusion ends with a Major Claim. This is great, it means your thesis statement is left in the reader's mind and is always a great way to close an essay.")
        for index in range(len(conclusion) -1):
            if index != 1 and index != len(conclusion)-1:
                if conclusion[index] == "MajorClaim":
                    feedback.append("Your conclusion contains a Major Claim, however it is not in a great position. Try to keep your Major Claims to the first or last sentence for better readability.")
    
    return feedback

def argumentative_to_none_argumentative_ratios(essay):
    total_paragraphs = essay["Total Paragraphs"].values[0]
    argumentative_to_non_argumentative_ratios = []
    
    for current_paragraph in range(2, total_paragraphs + 1):
        paragraph = essay.loc[(essay['Paragraph Number'] == current_paragraph)]
        paragraph_non_argumentative_count = 1
        paragraph_argumentative_count = 1
        for index, row in paragraph.iterrows():
            if row['Argument Component Type'] == "None":
                paragraph_non_argumentative_count += 1
            else:
                paragraph_argumentative_count += 1
        argumentative_to_non_argumentative_ratios.append(paragraph_argumentative_count/paragraph_non_argumentative_count)
        
        
    return argumentative_to_non_argumentative_ratios

def argumentative_to_none_argumentative_feedback(essay):
    ratio_list = argumentative_to_none_argumentative_ratios(essay)
    
    feedback = []
    for index in range(len(ratio_list)):
        if index == 0: #if introduction
            feedback.append("Your introduction has a ratio of " +  str(ratio_list[index]) + " of Argumentative to Non-Argumentative Sentences.")
            if ratio_list[index] > 0.25:
                feedback.append("This is decent - you are not diluting your introduction with sentences that do not really contribute to the overall message.")
            else:
                feedback.append("This is poor - in the introduction try to stay brief and on point so you can get to your main points sooner.")
        elif index == len(ratio_list)-1: #if conclusion
            feedback.append("Your conclusion has a ratio of" +  str(ratio_list[index])  + " of Argumentative to Non-Argumentative Sentences.")
            if ratio_list[index] > 0.25:
                feedback.append("This is decent - you are not diluting your conclusion with sentences that do not really contribute to the overall message.")
            else:
                feedback.append("This is poor - in the conclusion aim to summarise your overall thesis and not dilute the message.")
        else: #any other paragraph
            feedback.append("Paragraph " + str(index+1) + " has a ratio of"+ str(ratio_list[index]) + " of Argumentative to Non-Argumentative Sentences.")
            if ratio_list[index] > 1:
                feedback.append("This is good, in main body paragraphs we need to be introducing the bulk of our justifications so non-argumentative sentences may make our arguments less clear.")
            else:
                feedback.append("This is poor - while main body paragraphs are larger and more diverse, they still need to be focussed on creating sub-arguments to aid the overall thesis statements in the introduciton and conclusion.")

    return feedback

def main():
    train = pd.read_pickle("./train.pkl")
    test = pd.read_pickle("./test.pkl")

    test_essay_id = 4
    test_essay = test.loc[(test['Essay ID'] == test_essay_id)]

    #component_count_feedback(train, test_essay)
    #paragraph_component_count_feedback(train, test_essay)
    #paragraph_flow_feedback(test_essay)
    #argumentative_to_none_argumentative_feedback(test_essay)


# In[ ]:




