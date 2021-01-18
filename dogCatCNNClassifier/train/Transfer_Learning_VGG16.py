#TRANSFER LEARNING ON VGG-16
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers


#from the output above, we saw that block5_pool is the last/top layer of the vgg16 we have so we will add our layers from that point
vgg_model=VGG16(include_top=False, input_shape=(64,64,3))
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
#for layer in vgg_model.layers:
#  layer.trainable=False

x = layer_dict['block5_pool'].output
#add flatten layer so we can add the fully connected layer later
x = Flatten()(x)
#Fully connected layer
#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.2)(x)
#x = Dense(1024, activation='relu)(x)
#x = Dropout(0.2)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
#this is the final layer so the size of output in this layer is equal to the number of class in our problem
x = Dense(1, activation='sigmoid')(x)
#create the new model(recently the Model parameters were changed from input to inputs and output changed to outputs so it's now inputs = vgg16_model.input)
model = Model(inputs=vgg_model.input, outputs=x)
#print(model.summary())

import pandas as pd
dataset=pd.read_csv('../dataset/theFinalDataset.csv')
image_array = []
classLabels=[]

for j in range(0, 10000):
 pyString=dataset.iloc[j].tolist()
 classLabels.append(np.array(pyString[0][-1]))
 for index, item in enumerate(pyString):
        data = np.zeros((64,64), dtype=np.uint8)
        pixel_data = item.split()
        for i in range(0, 64):
            pixel_index = i * 64
            data[i] = pixel_data[pixel_index:pixel_index + 64]
        normData=[x/255 for x in data]
        normData=np.asarray(normData)
        normData=np.repeat(normData[:, :, np.newaxis],3,axis=2)
        image_array.append(np.array(normData))
        
        

image_array = np.array(image_array)
classLabels=[int(x) for x in classLabels ]
classLabels=np.asarray(classLabels)

x_train, y_train, x_test, y_test = [], [], [], []
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_array, classLabels, test_size=0.2, random_state=42)

x_train=x_train.reshape(x_train.shape[0],64,64,3)
x_test=x_test.reshape(x_test.shape[0], 64,64,3)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


gen = ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

#ADAMAX WORKS BEST 
model.compile(optimizer='adamax', loss="binary_crossentropy", metrics=["accuracy"])

print(y_train)
c_dogs=0
c_cats=0
for i in range(0,len(y_train)):
  if(y_train[i]==0):
    c_dogs=c_dogs+1
  else:
    c_cats=c_cats+1
print("Dogs: ", c_dogs, "\tCats: ", c_cats)

train_generator = gen.flow(x_train, y_train, batch_size=128)
#New Tensorflow update steps_per_epoch = length of x_train divided by the batch size
history=model.fit_generator(train_generator, steps_per_epoch=len(x_train)//128, epochs=37)
#model.fit(x_train, y_train, batch_size=64, epochs=25)
train_score = model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])

from sklearn.metrics import confusion_matrix
ANSWER=model.predict(x_test)
for i in range(0, len(ANSWER)):
  if(ANSWER[i]>=0.5):
    ANSWER[i]=1
  else:
    ANSWER[i]=0
print(ANSWER)
results=confusion_matrix(y_test, ANSWER)
print(results)

from sklearn.metrics import classification_report 
print(classification_report(y_test, ANSWER))

model_json = model.to_json()
with open("../model/modelVGG16.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model/modelweightsVGG16.h5")

