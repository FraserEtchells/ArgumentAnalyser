from flask import Flask, request, render_template
from Component_Classification_Features import *
from Component_Identification_Features import *
from Feedback_System import *
import os
import pandas as pd
import nltk



app = Flask(__name__)
dir = os.path.dirname(__file__)
pickled_scripts_folder = os.path.join(dir, 'PickledScripts')
list_of_pos_tags = [',','.',':','``',"''",'CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
nltk.download('punkt')

@app.route("/")
def form():
    return render_template('essay_submission_form.html')
    
@app.route('/', methods=['POST'])
def form_process():
    text = request.form['text']

    essay_dataframe = data_preprocess(text)
    column_names = essay_dataframe.columns.to_list()
    
    component_identification(essay_dataframe)
    component_classification(essay_dataframe)

    data_file =  open(os.path.join(pickled_scripts_folder,'train.pkl'), "rb") 
    data = pickle.load(data_file)
    data_file.close()

    component_count_feedback_list = component_count_feedback(essay_dataframe)
    paragraph_count_feedback_list = paragraph_component_count_feedback(essay_dataframe)
    paragraph_flow_feedback_list = paragraph_flow_feedback(essay_dataframe)
    argumentative_sentence_feedback_list = argumentative_to_none_argumentative_feedback(essay_dataframe)
    
    return render_template('essay_feedback.html', overall=component_count_feedback_list, paragraph_components = paragraph_count_feedback_list, paragraph_flows = paragraph_flow_feedback_list, argumentative = argumentative_sentence_feedback_list)

def component_identification(essay):
    copy_of_essay = essay.copy()
    position_features(copy_of_essay)
    token_features(copy_of_essay)
    similarity_features(copy_of_essay)

    model_file = open(os.path.join(pickled_scripts_folder,'component_identification_model.pickle'), "rb") 
    model = pickle.load(model_file)

    tfidf_file = open(os.path.join(pickled_scripts_folder,'tfidf.pickle'), "rb")
    tfidf = pickle.load(tfidf_file)

    pos_encoder_file = open(os.path.join(pickled_scripts_folder,'pos_encoder.pickle'), "rb")
    pos_encoder = pickle.load(pos_encoder_file)

    model_file.close()
    tfidf_file.close()
    pos_encoder_file.close()

    feature_columns=['Paragraph Number', 'Sentence', 'Sentence Similarity To Prompt', 'Most Common POS Token']
    essay_featurised = copy_of_essay.loc[:, feature_columns]

    essay_sentences = essay_featurised['Sentence']
    essay_sentences_vectorized = tfidf.transform(essay_sentences)
    essay_vectorized_dataframe = pd.DataFrame(essay_sentences_vectorized.todense(), columns=tfidf.get_feature_names())
    essay_concat = pd.concat([essay_featurised, essay_vectorized_dataframe], axis=1)
    essay_final = essay_concat.drop(['Sentence'], axis=1)

    essay_pos_encoded = pos_encoder.transform(copy_of_essay['Most Common POS Token'])
    essay_final['Most Common POS Token'] = essay_pos_encoded

    predictions = model.predict(essay_final)
    essay["Argumentative Label"] = predictions


def component_classification(essay):
    
    copy_of_essay = essay.copy()
    
    non_argumentative_sentences = copy_of_essay.index[copy_of_essay["Argumentative Label"] == 0]
    copy_of_essay.drop(non_argumentative_sentences, inplace = True)
    copy_of_essay.reset_index(drop=True, inplace=True)

    tokenisation_features(copy_of_essay)
    part_of_speech_features(copy_of_essay)
    positional_features(copy_of_essay)
    first_person_indicators_features(copy_of_essay)
    forward_indicator_feature(copy_of_essay)
    backward_indicator_feature(copy_of_essay)
    thesis_indicator_feature(copy_of_essay)

    model_file = open(os.path.join(pickled_scripts_folder,'component_classification_model.pickle'), "rb") 
    model = pickle.load(model_file)

    tfidf_file = open(os.path.join(pickled_scripts_folder,'tfidf_lemmatized.pickle'), "rb")
    tfidf = pickle.load(tfidf_file)

    model_file.close()
    tfidf_file.close()

    feature_columns=['Lemmatized Sentence','Paragraph Number', 'Sentence Within Introduction', 'Sentence Within Conclusion', 'Number of Proceeding Components', 'Number of Preceding Components' , 'First Person Indicator Present', 'First Person Indicator Count', 'Forward Indicator Present', 'Backward Indicator Present', 'Thesis Indicator Present']
    for curr_tag in list_of_pos_tags:
        feature_columns.append("Distribution of " + curr_tag + " POS Tag")
    essay_featurised = copy_of_essay.loc[:, feature_columns]

    essay_sentences = essay_featurised['Lemmatized Sentence']

    essay_sentences_vectorized = tfidf.transform(essay_sentences)
    essay_vectorized_dataframe = pd.DataFrame(essay_sentences_vectorized.todense(), columns=tfidf.get_feature_names())
    essay_concat = pd.concat([essay_featurised, essay_vectorized_dataframe], axis=1)
    essay_final = essay_concat.drop(['Lemmatized Sentence'], axis=1)

    predictions = model.predict(essay_final)
    predictions_list = predictions.tolist()
    for index in non_argumentative_sentences:
        predictions_list.insert(index, "None")

    for index, component in enumerate(predictions_list):
        if component == 0:
            predictions_list[index] = "MajorClaim"
        elif component == 1:
            predictions_list[index] = "Claim"
        elif component == 2:
            predictions_list[index] = "Premise"

    essay["Argument Component Type"] = predictions_list

def data_preprocess(essay_text):
    end_dataframe = pd.DataFrame(columns = ['Essay ID','Sentence', 'Source Paragraph', 'Paragraph Number'])
    curr_essay_id = 1
    raw_text = essay_text
        #Splits the raw data into paragraphs
    paragraphs = []
    for splits in raw_text.splitlines():
        if (not splits):
            continue
        elif (splits == "    "):
            continue
        else:
            paragraphs.append(splits)
                
    number_of_paragraphs = len(paragraphs)
        #Splits paragraphs into sentences. We do this rather than splitting the raw data into sentences as some essay prompts are not recognised by the system as sentences, so the prompt is often appended to the first sentence
    sentences = []
    for curr_paragraph in paragraphs:
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentences + tokenizer.tokenize(curr_paragraph)
            
    copy_sentences = sentences.copy()
        
    source_paragraph = []
    paragraph_numbers = []
    paragraph_iterator = -1
    sentence_iterator = -1
    for curr_paragraph in paragraphs:
        paragraph_iterator +=1
        sentence_iterator = -1
        for curr_sentence in copy_sentences:
            sentence_iterator +=1
            if (curr_paragraph.find(curr_sentence) != -1):
                source_paragraph.append(curr_paragraph)
                paragraph_numbers.append(paragraph_iterator + 1)
                #There was an issue where occasionally sentences appeared as sub-sentences in other sentences. For example "Physical Exercise" appeared in the sentence "Physical Exercises ..." which resulted in duplicate sentences.
                #By changing the current essay to this string, we ensure it will not be found multiple times (except for fringe cases but no essay should ever have the sentence AAAAAAAAAAAAA in it.)
                copy_sentences[sentence_iterator]= "AAAAAAAAAAAAAAAAAAAAAAAA"

    data = pd.DataFrame(columns = ['Essay ID','Sentence', 'Source Paragraph', 'Paragraph Number', 'Total Paragraphs'])
    for i in range(len(sentences)):
        data = data.append({'Essay ID': curr_essay_id, 'Sentence':sentences[i],'Source Paragraph':source_paragraph[i], 'Paragraph Number':paragraph_numbers[i], 'Total Paragraphs':number_of_paragraphs}, ignore_index=True)
    return data
    

