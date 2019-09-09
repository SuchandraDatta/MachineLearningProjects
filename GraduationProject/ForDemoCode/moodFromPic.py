import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image

#-----------------------------
#opencv initialization
def emotion_analysis(emotions):
 objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
 y_pos = np.arange(len(objects))
 
 plt.bar(y_pos, emotions, align='center', alpha=0.5)
 plt.xticks(y_pos, objects)
 plt.ylabel('percentage')
 plt.title('emotion')
 
 plt.show()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#cap = cv2.VideoCapture('friends.mkv')
#-----------------------------
#face expression recognizer initialization
from keras.models import model_from_json
#Put path name to json subject to OS being used
model = model_from_json(open("path_to_file/myMoodModel.json", "r").read())
model.load_weights('path_to_file/model_weights.h5') #load weights
print(model.summary())
#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

img = cv2.imread('me.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#print(faces) #locations of detected faces

for (x,y,w,h) in faces:
 cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
 detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
 detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
 detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
		
 img_pixels = image.img_to_array(detected_face)
 img_pixels = np.expand_dims(img_pixels, axis = 0)
		
 img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
 predictions = model.predict(img_pixels) #store probabilities of 7 expressions
		
		#find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
 max_index = np.argmax(predictions[0])
 emotion = emotions[max_index]
		
		#write emotion text above rectangle
 cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
 cv2.imshow('img',img)
 emotion_analysis(predictions[0])
cv2.waitKey(0)
cv2.destroyAllWindows()