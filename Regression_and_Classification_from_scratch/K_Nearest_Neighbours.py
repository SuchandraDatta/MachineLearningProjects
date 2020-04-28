#K nearest neighbours algorithm
import numpy as np
import matplotlib.pyplot as plt

def EuclideanDistance(new_point, datasetX):
  sum_squares=[]
  for i in range(0, len(new_point)):
    sum_squares.append((new_point[i]-datasetX[i])**2);
  sum_squares=np.asarray(sum_squares)
  ans=np.sqrt(np.sum(sum_squares))
  return ans

#Standardization performed
def feature_scaling(on_this_array):
  if(len(on_this_array.shape)==2):
    on_this_array=on_this_array.astype(np.double)
    for i in range(0, len(on_this_array[0])):
        meanValue=np.mean(on_this_array[:,i])
        stdValue=np.std(on_this_array[:,i])
        on_this_array[:,i]=(on_this_array[:,i]-meanValue)/stdValue
  else:
    meanValue=np.mean(on_this_array)
    stdValue=np.std(on_this_array)
    on_this_array[:]=(on_this_array[:]-meanValue)/stdValue
  return on_this_array

datasetX=[
         [7,7], [7,4],[3,4],[1,4]
]
datasetY=[0,0,1,1]
datasetX=np.asarray(datasetX)
datasetY=np.asarray(datasetY)

datasetX=feature_scaling(datasetX)
new_point=[3,7]
new_point=np.asarray(new_point)
new_point=feature_scaling(new_point)
#Get the Euclidean distance on new point from all points in the dataset
distance_list=[]
for i in range(0, len(datasetX)):
  distance_list.append(EuclideanDistance(new_point, datasetX[i]))

#Select k sort distances, select least k distances, mode of the class labels for least k distances is the required class
#For regression, the mean of the least k values is the required prediction
k=3
least=0
labels_to_consider=[]
copyY=datasetY
print(distance_list)
for i in range(0, len(distance_list)):
  for j in range(0, len(distance_list)-1-i):
    if(distance_list[j]>=distance_list[j+1]):
      temp=distance_list[j]
      distance_list[j]=distance_list[j+1]
      distance_list[j+1]=temp
      temp=copyY[j]
      copyY[j]=copyY[j+1]
      copyY[j+1]=temp
print(distance_list)
print(copyY)

labels_to_consider=copyY[0:k]
print(labels_to_consider)
from scipy import stats
label_selected=stats.mode(labels_to_consider)
print("The label selected is: ", label_selected[0][0])