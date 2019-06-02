import pandas as pd
import numpy as np
import matplotlib.pyplot
import csv

dfToList=[]
filenames=['amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt']
for i in range(0,3):
  f=pd.read_csv(filenames[i],  names=['review', 'label'], delimiter="\t");
  dfToList.append(f);
  
 
datasetArray=[];
for i in range(0, 1):
	for j in range(0, 1000):
		datasetArray.append(np.asarray(dfToList[i].iloc[j][:]))

for i in range(1, 2):
	for j in range(0, 748):
		datasetArray.append(np.asarray(dfToList[i].iloc[j][:]))
for i in range(2, 3):
	for j in range(0, 1000):
		datasetArray.append(np.asarray(dfToList[i].iloc[j][:]))

datasetArray=np.asarray(datasetArray)
print(datasetArray.shape)
X=datasetArray[0:2747, 0]
X=X.reshape(1,2747)
Y=datasetArray[0:2747,1]

print("\nX shape\t Y shape\n")
print(X.shape)
print(Y.shape)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range(0, 2747):
 review=re.sub('[^a-zA-Z]', ' ', X[0][i]) #remove unwanted symbols
 review=review.lower() #convert to lowercase
 review=review.split() #split to words, list of words output
 ps=PorterStemmer()
 review=[ ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #removing words not a stopword
 #set works faster as py goes through set faster than list
 review=' '.join(review) #joining words together sep by spaces
 corpus.append(np.asarray(review))

X=np.asarray(corpus)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=42)

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=6600)
tokenizer.fit_on_texts(x_train)

X_train = tokenizer.texts_to_sequences(x_train)
X_test = tokenizer.texts_to_sequences(x_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

from keras.preprocessing.sequence import pad_sequences

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(X_train.shape)
print(X_test.shape)
from keras.models import Sequential
from keras import layers

embedding_dim = 158

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
model.add(layers.Conv1D(128, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
                    epochs=12,
                    verbose=True,
                    #validation_data=(X_test, y_test),
                    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: " , accuracy*100 )
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: " , accuracy*100)


