import re
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, util

modelPath = "../models/sent_trans"
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# model.save(modelPath)
model = SentenceTransformer(modelPath)
# print('model saved')
def question_search(ques, embedding_df):
    """
    This function return the matching values to the question from the passed dataframe
    :param ques: str
    :param embedding_df: dataframe
    :return series
    """

#     ques = [ques]
    sentence_embd = embedding_df["que_embd"].to_list()
#     print(sentence_embd)
    question_embd = model.encode(ques, convert_to_tensor=True)
#     print(question_embd)
    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(sentence_embd, question_embd)
    embedding_df["score"] = [round(float(score),2) for score in cosine_scores] # remove decimal points
    result = embedding_df[embedding_df["score"]>=0.50]['Questions'].reset_index(drop=True)
    return result

# define a function to clean the text
def clean_text(text):
    """
    This function returns clean lowercase text after removing all characters other than the alphabets
    :param text: str
    :return str
    """
    temp = re.sub('[^a-zA-Z]', ' ', text)
    temp = temp.lower()
    temp = temp.split()
    temp = ' '.join(temp)
    return temp

# define a function to clean up the input from user and tokenize it for inference
def clean_up_sentence(sentence):
    """
    This function tokenizes sentence using nltk library
    :param sentence: str
    :return sentence_words: list
    """
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    """
    This function represents words in documents in sparse matrix format
    :param sentence: str
    :param words: list
    :param show_details: bool
    :return numpy array
    """
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

    
    