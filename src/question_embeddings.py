from sentence_transformers import SentenceTransformer, util
import pandas as pd

# load the pretrained encoder model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# load the question-answer dataset
df = pd.read_csv('../input/encoder_data.csv')

# fit the encoder to question column
que_encoding = model.encode(df["Questions"].to_list())

# append the embeddings for the questions in the dataframe
temp = [list(i)  for i in que_encoding]
df["que_embd"] = temp

# save the embeddings for later use
df.to_csv('../input/question_embd.csv',index=False)

print(df.head())
