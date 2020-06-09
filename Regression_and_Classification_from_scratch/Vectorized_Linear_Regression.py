#For calculations, X must be of shape(Number of features, Number of training samples) so given the dataset where the #observations are row-wise we take the transpose of it to get the observations column-wise. This helps in #vectorization and the bias must be taken as a separate term not part of the theta matrix
def predict(thetaParam,bias, x_params):
    x_params=np.asarray(x_params)
    x_params=x_params.T
    print("x_params=", x_params,"\ttheta=", thetaParam)
    ans=np.dot(thetaParam, x_params)+bias
    print("ans=", ans)
    return ans
  
import numpy as np
x=[[0,0], [1,1], [2,3],[1,3] ]
y=[6,16,30,24]

x=np.asarray(x)
y=np.asarray(y)
print("X=", x)
print("Y=", y)

flag=0
theta=np.random.rand(x.shape[1])
print("Theta=", theta)
bias=np.random.rand(1)
lambda_term=0
print("Bias=", bias)
x=x.T
m=x.shape[1]
alpha=0.01
while(True):
    flag=0
    step1=np.dot(theta,x)+bias
    #h(x)-y
    step2=step1-y
    step2=np.square(step2)
    #print("After squaring step2: ", step2)
    cost_function=np.sum(step2)
    #print("Cost function part 1: ", cost_function)
    cost_function=(cost_function)/(2*m)+lambda_term*(np.sum(np.square(theta)))/(2*m)
    print("Cost function with R2 regularization term: ", cost_function)

    step3=np.dot(x, (step1-y))
    #print(step3)
    change_theta=theta*(1-(alpha*lambda_term)/m)-(alpha/m)*step3
    change_bias=bias-(alpha/m)*np.sum((step1-y))
    hold_array=np.abs(theta-change_theta)
    #print("change_theta=", change_theta)
    #print("change_bias=", change_bias)
    '''for i in range(0, len(change_theta)):
            hold=theta[i]-change_theta[i]
            if(hold<0):
                hold=-hold
            if(hold>=1e-15):
                flag=1
                break
    if(flag==0):
        print("Converged")
        break'''
    if(hold_array.any()>=1e-15):
        flag=1
    if(flag==0):
        print("Converged")
        break
    theta=change_theta
    bias=change_bias
    #dummy=input("Give input")
print("THETA=", theta)
print("BIAS=", bias)

from sklearn.metrics import r2_score
r2score_hell=r2_score([16,36,40,72], predict(theta,bias,[[1,1], [1,6], [1,7], [4,8]]))
print("r2 score for test set linear regression: ", r2score_hell)