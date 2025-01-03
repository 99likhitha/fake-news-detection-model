import nltk
nltk.download('stopwords')
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st


df=pd.read_csv("train.csv")
df=df.fillna(' ')
df['news']=df['title']+" "+df['author']
ps=PorterStemmer()
def stemming(news):
    stemmed_news=re.sub('[^a-zA-z]'," ",news) #subtract all other chars except alphabets
    stemmed_news=stemmed_news.lower() #convert them to lower case
    stemmed_news=stemmed_news.split() #split the words
    stemmed_news=[ps.stem(word) for word in stemmed_news if not word in stopwords.words('english')]  #use each word in this base tense form
    return " ".join(stemmed_news) #join words by spaces

df['news']=df['news'].apply(stemming)
X=df['news'].values
Y=df['label'].values

vector = TfidfVectorizer()
vector.fit(X)
X=vector.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)
model=LogisticRegression()
model.fit(X_train,y_train)
# y_pred_on_train=model.predict(X_train)

# y_pred_on_test=model.predict(X_test)



#Streamlit 
st.title("Fake news Prediction")
input_text=st.text_input("Enter article you want to check")


def predict(input_text):
    input_data=vector.transform([input_text])
    prediction=model.predict(input_data)
    return prediction[0]

if input_text:
    pred=predict(input_text)
    if pred ==1:
        st.write("The article/news is Fake")
    else:
        st.write("The article/news is Real")

         
