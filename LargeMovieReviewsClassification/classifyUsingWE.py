import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('theFinalDatasetMovieReviews.csv', names=['reviews'])
x=[]
y=[]
for i in range(0, 49968):
  pyString=dataset['reviews'][i]
  y.append(np.asarray(int(pyString[-1])))
  x.append(np.asarray(pyString))
x=np.asarray(x)
y=np.asarray(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

from keras.preprocessing.text import Tokenizer
tokenOBJ=Tokenizer(num_words=9000)
tokenOBJ.fit_on_texts(x_train)

x_train=tokenOBJ.texts_to_sequences(x_train)
x_test=tokenOBJ.texts_to_sequences(x_test)

voc_size=len(tokenOBJ.word_index)+1

from keras.preprocessing.sequence import pad_sequences
maxlength=100
x_train=pad_sequences(x_train, padding='post', maxlen=maxlength)
x_test=pad_sequences(x_test, padding='post', maxlen=maxlength)

print(x_train.shape)
print(x_test.shape)

from keras.models import Sequential
from keras import layers

embedDIM=100

model=Sequential()
model.add(layers.Embedding(input_dim=voc_size, output_dim=embedDIM, input_length=maxlength))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
history=model.fit(x_train, y_train, epochs=1, verbose=True, batch_size=4000)

loss, acc1=model.evaluate(x_train, y_train)
print("TRAIN LOSS: " , loss, "\tTRAIN ACCURACY: ", acc1*100)
loss, acc2=model.evaluate(x_test, y_test)
print("TEST LOSS: " , loss, "\tTEST ACCURACY: ", acc2*100)
