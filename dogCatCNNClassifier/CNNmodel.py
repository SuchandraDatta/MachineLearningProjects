from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from keras.layers import Dense, Dropout, Input, Flatten
from keras import optimizers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

model=Sequential();

#One block
model.add(Conv2D(64, (3,3), input_shape=(64,64,1), activation="relu"));
model.add(MaxPooling2D((3,3)))

#Block two
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D((3,3)))

#Block three
model.add(Conv2D(256, (3,3), activation="relu"))
model.add(MaxPooling2D((3,3)))

model.add(Flatten())

#Final connections
model.add(Dense(activation="relu", units=256))
model.add(Dropout(0.2))
model.add(Dense(activation="sigmoid", units=1))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary());

#Getting the data
import pandas as pd
dataset=pd.read_csv('theFinalDataset.csv')
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
        #VERY VERY IMPORTANT STEP normalisation
        normData=[x/255 for x in data]
        image_array.append(np.array(normData))
        
image_array = np.array(image_array)
classLabels=np.asarray(classLabels)
x_train, y_train, x_test, y_test = [], [], [], []
x_train, x_test, y_train, y_test = train_test_split(image_array, classLabels, test_size=0.2, random_state=42)

x_train=x_train.reshape(x_train.shape[0],64,64,1)
x_test=x_test.reshape(x_test.shape[0], 64,64,1)


gen = ImageDataGenerator(featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

train_generator = gen.flow(x_train, y_train, batch_size=256)
history=model.fit_generator(train_generator, steps_per_epoch=256, epochs=75)

train_score = model.evaluate(x_train, y_train, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])
 
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])
  
model_json = model.to_json()
with open("model/myMoodModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model/model_weights.h5")






