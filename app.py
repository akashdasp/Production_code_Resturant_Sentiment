import streamlit as st
import pickle
import string

from nltk_download_utils import stopwords,PorterStemmer 
import json
ps=PorterStemmer()

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]

  for i in text:
    if i.isalnum():
      y.append(i)
  text=y[:]
  y.clear()

  for i in text:
    if i  not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text=y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)
tfidf=pickle.load(open("vectorizer.pkl", 'rb'))
with open('model_cat.pkl', 'rb') as f:
    model = pickle.load(f)
st.title("Resturant Reviews Classifier ?")
input_sms=st.text_area("Enter The Message")

if st.button("Predict"):

  # 1 .preprocess
  tranform_sms=transform_text(input_sms)
  # 2.vectorize
  vector_input=tfidf.transform([tranform_sms])
  # 3.predict
  result=model.predict(vector_input)
  # 4.Display
  if result==1:
      st.header("Postive")
  else:
      st.header("Negetive")