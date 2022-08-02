import pandas as pd
from utils import question_search

df = pd.read_csv("../input/question_embd.csv")

df["que_embd"] = df["que_embd"].apply(lambda x:eval(x))

while True:
    texts = input(str('input the text: '))
    if texts == '/stop':
        break
    else:
        res = question_search(texts, df)
        if res.empty:
            print('Sorry, I am unable to understand. Kindly rephrase your question!')
        else:
            print(df[df['Questions']==res[0]]['Answers'])