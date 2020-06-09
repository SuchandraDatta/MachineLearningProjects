#For calculations, X must be of shape(Number of features, Number of training samples) so given the dataset where the #observations are row-wise we take the transpose of it to get the observations column-wise. This helps in #vectorization and the bias must be taken as a separate term not part of the theta matrix
from scipy.special import expit

def predict(thetaParam,bias, x_params):
    x_params=np.asarray(x_params)
    x_params=x_params.T
    print("x_params=", x_params,"\ttheta=", thetaParam)
    ans=np.dot(thetaParam, x_params)+bias
    print("ans=", ans)
    return ans
  
import numpy as np
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
print("X=", x)
print("Y=", y)

flag=0
theta=np.random.rand(x.shape[1])
theta=np.asarray(theta)
bias=np.random.rand(1)
lambda_term=0
x=x.T
m=x.shape[1]
alpha=0.01
while(True):
    flag=0
    #h(x)
    step1=np.dot(theta,x)+bias
    step1=expit(step1)#To apply sigmoid function to every term of step1
    #ylog(h(x))-(1-y)log(1-h(x))
    step2=0
    step2=step2-y*np.log(step1)-(1-y)*np.log(1-step1)
    #If the value of step1 or 1-step1 becomes zero(which it does after lots of iterations and nearer the minimum, log(0)=math error -inf so we break the loop)
    if(np.isnan(np.sum(step2))):
        break
     
    cost_function=np.sum(step2)
    #print("Cost function part 1: ", cost_function)
    cost_function=(cost_function)/(2*m)+lambda_term*(np.sum(np.square(theta)))/(2*m)
    print("Cost function with R2 regularization term: ", cost_function)

    step3=np.dot(x, (step1-y))
    #print(step3)
    change_theta=theta*(1-(alpha*lambda_term)/m)-(alpha/m)*step3
    change_bias=bias-(alpha/m)*np.sum((step1-y))
    hold_array=np.abs(theta-change_theta)
    if(hold_array.any()>=1e-05):
        flag=1
    #print("change_theta=", change_theta)
    #print("change_bias=", change_bias)
    if(flag==0):
        print("Converged")
        break
    theta=change_theta
    bias=change_bias
    
print("THETA=", theta)
print("BIAS=", bias)
#The actual answer is THETA= [ 4.27105379 -6.5152645 ]
#BIAS= [-1.89351873]