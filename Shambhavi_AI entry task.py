import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
plt.rcParams['figure.figsize'] = (12.0, 9.0)

df = pd.read_csv("Boston.csv")

X = df[['crim','zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',	'ptratio', 'black', 'lstat']]

Y = df['medv']

X=X.to_numpy()
Y=Y.to_numpy()
#pandas extract data in dataframe format but to perform matrix operations on these data it should be first
#converted into numpy matrix
sz=X.shape

nic=sz[1]
n = float(len(df['crim']))

X_mn=np.mean(X,axis=0)
X_sd=np.std(X,axis=0)
X=(X-X_mn)/X_sd
#circular normalization of data in X to get a better fitting plot
Y_mn=np.mean(Y)
Y_sd=np.std(Y)
Y=(Y-Y_mn)/Y_sd
#circular normalisation of data in Y

m=np.zeros([nic, 1],dtype=float)
D_m=np.zeros([nic, 1],dtype=float)
c = 0
D_c=0

L = 0.001
iterations = 100000
n = float(len(df['crim']))
for i in range(iterations):
    
    Yf=sum(numpy.transpose(X)*m)+c #predicted values of Y
    kp=0
    for j in range(nic):
        D_m[kp,0]=(-2/n)*sum(X[:,kp]*(Y-Yf))
        #first differenciation of the cost function with respect to slope of each independent variable
        m[kp,0]=m[kp,0]-L*D_m[kp,0]
        #updating the values of m after each iteration
        kp=kp+1
    D_c=(-2/n)*sum(Y-Yf)
    #differenciation of cost function with respect to c
    c=c-L*D_c
    #updating value of c

print(m)    
print(c)

kp=0
for i in range(nic-1):
    plt.subplot(3,4,kp+1)
    plt.scatter(X[:,kp],Y, s=2)
    plt.plot(X[:,kp],X[:,kp]*m[kp]+c,color='red')
    kp=kp+1
plt.show()    

#The output consists of a matrix of (13,1) which is the value of m for 13 independent variable and the final value of c

#HackerRank ID = @sinhashambhavi11


