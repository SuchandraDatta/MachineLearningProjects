from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import model_from_json
#from keras.models import model_from_json
model = model_from_json(open("../model/modelResNet50.json", "r").read())
model.load_weights('../model/modelweightsResNet50.h5') #load weights
#print(model.summary()) 

from PIL import Image
import numpy as np
image=Image.open('../test_images/penguin.jpg').resize((64,64)).convert('L')
image=np.asarray(image)
image=image.flatten()
image=[x for x in image]#No normalization as it wasn't done during training
image=np.reshape(image, (64,64))
#Copy grayscale 3 times in each of the color channels, not much adversely affecting the performance
image=np.repeat(image[:,:,np.newaxis], 3, axis=2)

print(image.shape)
image=image.reshape(1,64,64,3)
print(image.shape)
#Output y=0 to 0.5 should be a dog and 0.5 upwards to 1 should be a cat
y=model.predict(image)
print("CHECKING RESNET 50")
print(y)
if(y>=0.5):
    print("IT'S A CAT")
else:
    print("IT'S A DOG")
