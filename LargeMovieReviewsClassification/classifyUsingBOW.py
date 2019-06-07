import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('theFinalDatasetMovieReviews.csv', names=['reviews'])

#Preprocess the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
y=[]

for i in range(0, 49968):
 pyString=dataset['reviews'][i]
 y.append(np.asarray(int(pyString[-1])))
 sentence=re.sub('[^a-zA-Z]', ' ', dataset['reviews'][i])
 sentence=sentence.lower()
 sentence=sentence.split()  
 obj=PorterStemmer()  
 sentence=[ obj.stem(word) for word in sentence if not word in set(stopwords.words('english')) ]
 sentence=' '.join(sentence)
 corpus.append(sentence)
 

y=np.asarray(y)
from sklearn.feature_extraction.text import CountVectorizer
obj=CountVectorizer(max_features=1500)
x=obj.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train, y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

print(cm)

