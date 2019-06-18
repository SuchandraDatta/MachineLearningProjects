import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def NB_classifier():
	from sklearn.naive_bayes import GaussianNB
	classifier=GaussianNB()
	return classifier
def LR_classifier():
	from sklearn.linear_model import LogisticRegression
	classifier=LogisticRegression(solver='liblinear', random_state=0)
	return classifier
def SVM_classifier():
	from sklearn.svm import SVC
	classifier=SVC(kernel='linear', random_state=0)
	return classifier
def KNN_classifier():
    from sklearn.neighbors import  KNeighborsClassifier
    classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2 )
    return classifier
def DecTree_classifier():
    from sklearn.tree import  DecisionTreeClassifier
    classifier=DecisionTreeClassifier(random_state=0)
    return classifier
def RandomForest_classifier():
    from sklearn.ensemble import  RandomForestClassifier
    classifier=RandomForestClassifier(n_estimators=200, random_state=0)
    return classifier

def testing(classifier, y_test, x_test):
 y_pred = classifier.predict(x_test)
 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(y_test, y_pred)
 print(cm)
 testaccuracy=(cm[0][0]+cm[1][1])/200;
 testaccuracy=testaccuracy*100;
 print("Test accuracy: ", testaccuracy)

#IMPORT THE DATASET
dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) #Ignore double quotes
#Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]

for i in range(0, 1000):
 review=re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) #remove unwanted symbols
 review=review.lower() #convert to lowercase
 review=review.split() #split to words, list of words output
 ps=PorterStemmer()
 review=[ ps.stem(word) for word in review if not word in set(stopwords.words('english'))] #removing words not a stopword
 #set works faster as py goes through set faster than list
 review=' '.join(review) #joining words together sep by spaces
 corpus.append(review) 
#BOW model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1650)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values
print(X.shape)
#ML model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

#FOR NAIVE BAYES
print("RESULTS FOR NB")
classifier=NB_classifier()
classifier.fit(x_train, y_train)
testing(classifier, y_test, x_test)


#FOR LOGISTIC REGRESSION
print("RESULTS FOR LOGISTIC REGRESSION")
classifier=LR_classifier()
classifier.fit(x_train, y_train)
testing(classifier, y_test, x_test)

#FOR SVM
print("RESULTS FOR SVM")
classifier=SVM_classifier()
classifier.fit(x_train, y_train)
testing(classifier, y_test, x_test)

#FOR KNN
print("RESULTS FOR KNN")
classifier=KNN_classifier()
classifier.fit(x_train, y_train)
testing(classifier, y_test, x_test)

#FOR DECISION TREES
print("RESULTS FOR DECSION TREES")
classifier=DecTree_classifier()
classifier.fit(x_train, y_train)
testing(classifier, y_test, x_test)


#FOR RANDOM FORESTS
print("RESULTS FOR RANDOM FOREST")
classifier=RandomForest_classifier()
classifier.fit(x_train, y_train)
testing(classifier, y_test, x_test)

