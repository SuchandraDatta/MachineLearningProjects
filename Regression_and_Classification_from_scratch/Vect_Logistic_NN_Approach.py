import numpy as np
from scipy.special import expit

x=[
   [2.7810836,	2.550537003 ],
   [1.465489372,	2.362125076 ],
   [ 3.396561688,	4.400293529],
   [ 1.38807019,	1.850220317],
   [3.06407232,	3.005305973],
   [ 7.627531214,	2.759262235],
   [5.332441248,	2.088626775],
   [ 6.922596716,	1.77106367],
   [8.675418651,	-0.2420686549],
   [ 7.673756466,	3.508563011 ]
   ]
y=[0,0,0,0,0,1,1,1,1,1]
x=np.asarray(x)
y=np.asarray(y)
x=x.T
theta=np.random.rand(x.shape[0])
bias=np.random.rand(1)
alpha=1
flag=0
m=x.shape[1]
while(True):
   flag=0
   Z=np.dot(theta,x)+bias
   A=expit(Z)
   dZ=A-y
   if(np.isnan(np.sum(dZ))):
      break
   dW=(np.dot(x,dZ.T))/m
   dB=(np.sum(dZ))/m
   change_theta=theta-alpha*dW
   change_bias=bias-alpha*dB
   hold_array=np.abs(theta-change_theta)
   if(np.max(hold_array)>=1e-05):
      flag=1
   if(flag==0):
      print("Converged")
      break
   theta=change_theta
   bias=change_bias
   print("Theta=", theta)

print("Theta=", theta)
print("Bias=", bias)


                           