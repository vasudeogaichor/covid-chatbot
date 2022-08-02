from flask import Flask
app = Flask(__name__)
from flask import Flask, jsonify, request 
import numpy as np
import pandas as pd
from keras.models import load_model
import json
import pickle
from utils import bag_of_words
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from utils import question_search

df = pd.read_csv("../input/question_embd.csv")
df["que_embd"] = df["que_embd"].apply(lambda x:eval(x))

chatbot = load_model('../models/chatbot_model/chatbot_model.h5')
f = open('../input/common_intents.json')
intents = json.load(f)
words = pickle.load(open('../input/words.pkl','rb'))
classes = pickle.load(open('../input/classes.pkl','rb'))

def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = chatbot.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    if tag == 'covid_question':
        result = 'This is a coronavirus related question!'
    else:
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = i['responses'][0]
                break
            # if(i['tag']=='covid_question'):
            #     result = 'This is a coronavirus related question!'
    return result

def get_encoder_response(sentence):
    res = question_search(sentence, df)
    if res.empty:
        response = 'Sorry, I am unable to understand. Kindly rephrase your question!'
    else:
        response = df[df['Questions']==res[0]]['Answers']
        response = response.reset_index(drop=True)
    return response

@app.route('/predict', methods=['POST'])
def chatbot_reply():
    data = request.get_json(force=True)
    sentence = data['sentence']
    print(sentence)
    ints = predict_class(sentence)
    reply = getResponse(ints, intents)
    if reply != 'This is a coronavirus related question!':
        return jsonify(reply)
    else:
        encoder_response = get_encoder_response(sentence)
        print(encoder_response[0])
        return jsonify(encoder_response[0])
        

if __name__ == '__main__':
   app.run(debug = True)