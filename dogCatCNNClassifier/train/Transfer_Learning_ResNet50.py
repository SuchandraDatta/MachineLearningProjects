from keras.applications import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

resnet50_model=ResNet50(include_top=False, input_shape=(64,64,3))
layer_dict = dict([(layer.name, layer) for layer in resnet50_model.layers])

print(resnet50_model.summary())
x = resnet50_model.output
#add flatten layer so we can add the fully connected layer later
x = Flatten()(x)
#Fully connected layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
#this is the final layer so the size of output in this layer is equal to the number of class in our problem
x = Dense(1, activation='sigmoid')(x)
#create the new model #create the new model(recently the Model parameters were changed from input to inputs and output changed to outputs so it's now inputs = resnet50_model.input)
model = Model(inputs=resnet50_model.input, outputs=x)
print(model.summary())

import pandas as pd
dataset=pd.read_csv('../dataset/theFinalDataset.csv')
image_array = []
classLabels=[]

for j in range(0, 10000):
 pyString=dataset.iloc[j].tolist()
 classLabels.append(np.array(pyString[0][-1]))
 for index, item in enumerate(pyString):
        data = np.zeros((64,64), dtype=np.uint8)
        # split space separated ints
        pixel_data = item.split()
        for i in range(0, 64):
            pixel_index = i * 64
            data[i] = pixel_data[pixel_index:pixel_index + 64]

        normData=[x for x in data]#Remove the /255 normalization
        #np.repeat(a[:, :, np.newaxis], 3, axis=2)
        normData=np.asarray(normData)
        normData=np.repeat(normData[:, :, np.newaxis],3,axis=2)
        image_array.append(np.array(normData))
        
        

image_array = np.array(image_array)
classLabels=np.asarray(classLabels)
classLabels=[ int(x) for x in classLabels]

x_train, y_train, x_test, y_test = [], [], [], []
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(image_array, classLabels, test_size=0.2, random_state=42)

x_train=x_train.reshape(x_train.shape[0],64,64,3)
x_test=x_test.reshape(x_test.shape[0], 64,64,3)
#New Tensorflow changes, the y values MUST BE A NUMPY ARRAY can't be a list anymore
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

gen = ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
train_generator = gen.flow(x_train, y_train, batch_size=128)
history=model.fit_generator(train_generator, steps_per_epoch=128, epochs=12)
#model.fit(x_train, y_train, batch_size=256, epochs=20)
train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])

model_json = model.to_json()
with open("../model/modelResNet50.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model/modelweightsResNet50.h5")
