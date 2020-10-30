# BOSTONPRICEPREDICTION
#Using Regression ALgorithm
#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from IPython.display import HTML


# In[55]:


#load the dataset
boston=load_boston()

#description of the dataset
print(boston.DESCR)


# In[56]:


#Put the data into pandas dataframes
features=pd.DataFrame(boston.data,columns=boston.feature_names)
features


# In[57]:


features['AGE']


# In[58]:


target=pd.DataFrame(boston.target,columns=['target'])
target


# In[59]:


max(target['target'])


# In[60]:


min(target['target'])


# In[61]:


#Concatenate features and target into a single dataframe
# axis=1 makes it concatenate coulumn wise

df=pd.concat([features,target],axis=1)
df


# In[62]:


# use round(decimals=2) to set the precision to 2 decimal places
df.describe().round(decimals=2)


# In[63]:


#calculate correlation with every column of the data(with the target)
corr=df.corr('pearson')

#take absolute values of correlations
corrs=[abs(corr[attr]['target']) for attr in list(features)]

#make a list of pairs [(corr,feature)]
l=list(zip(corrs,list(features)))

#sort the list of pairs in ascending/descending order
#with the correaltion value as the key for sorting
l.sort(key=lambda x:x[0],reverse=True)

#unzip pairs into two lists
#zip(*l)-takes a list that looks like [[a,b,c],[d,e,f],[g,h,i]]
#and returns [[a,d,g],[b,e,h],[c,f,i]]
corrs,labels=list(zip((*l)))

#plot correlations with respect to the target variable as a bar graph
index=np.arange(len(labels))
plt.figure(figsize=(15,5))
plt.bar(index,corrs,width=0.5)
plt.xlabel('Attributes')
plt.ylabel('Correlation with the target variable')
plt.xticks(index,labels)
plt.show


# In[64]:


#NORMALIZE THE DATA WITH MINMAX SCALAR( BRING TO SAME UNITS -RANGE 0-1)


# In[65]:


X=df['LSTAT'].values
Y=df['target'].values


# In[66]:


#Before Normalization
print(Y[:5])


# In[67]:


x_scaler=MinMaxScaler()
X=x_scaler.fit_transform(X.reshape(-1,1))
X=X[:,-1]
y_scaler=MinMaxScaler()
Y=y_scaler.fit_transform(Y.reshape(-1,1))
Y=Y[:,-1]


# In[68]:


#After normalization
print(Y[:5])


# In[69]:


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


# In[19]:


#Generate n evenly spaced values from zero radians to  PI radians
n=200
x=np.linspace(0,2*np.pi,n)
sine_values=np.sin(x)

#Plot the sinwave
plt.plot(x,sine_values)


# In[70]:


#Add some noise to the sinewave
noise=0.5
noisy_sine_values=sine_values+np.random.uniform(-noise,noise,n)

#Plot the noisy values
plt.plot(x,noisy_sine_values,color='r')
plt.plot(x,sine_values,linewidth=3)


# In[71]:


#Calculate MSE using the equation
error_value=(1/n)*sum(np.power(sine_values-noisy_sine_values,2))
error_value


# In[72]:


#Calculate mean saured error using the function from the sklearn
#library
mean_squared_error(sine_values,noisy_sine_values)


# In[73]:


def error(m,x,c,t):
    N=x.size
    e=sum(((m*x+c)-t)**2)
    return e*1/(2*n)


# In[74]:


#SPLITTING DATA INTO FIXED SETS
#0.2 indicates that 20% of data is randomly sampled as testing data,x=features,y=target

xtrain,xtest,ytrain,ytest= train_test_split(X,Y,test_size=0.2)


# In[75]:


def update(m,x,c,t,learning_rate):
    grad_m=sum(2*((m*x+c)-t)*x)
    grad_c=sum(2*((m*x+c)-t))
    m=m-grad_m*learning_rate
    c=c-grad_c*learning_rate
    return m,c
    
#if the error falls below the threshold,the gradient descent process is stopped and the weights are returned


# In[76]:


def gradient_descent(init_m,init_c,x,t,learning_rate,iterations,error_threshold):
    m=init_m
    c=init_c
    error_values=list()
    mc_values=list()
    for i in range(iterations):
        e=error(m,x,c,t)
        if e<error_threshold:
            print('Error less than the threshold.Stopping gradient descent')
            break
        error_values.append(e)
        m,c=update(m,x,c,t,learning_rate)
        mc_values.append((m,c))
    return m,c,error_values,mc_values


# In[77]:


get_ipython().run_cell_magic('time', '', 'init_m=0.9\ninit_c=0\nlearning_rate=0.001\niterations=250\nerror_threshold=0.001\n\nm,c,error_values,mc_values=gradient_descent(init_m,init_c,xtrain,ytrain,learning_rate,iterations,error_threshold)')


# In[78]:


#ANIMATION
#As the number of iterations increases,changes in the line are less noticeable
#Inorder to reduce the processing time for the animation,it is advised to choose smaller values
mc_values_anim=mc_values[0:250:5]


# In[79]:


fig,ax=plt.subplots()
ln,=plt.plot([],[],'ro-',animated=True)


def init():
    plt.scatter(xtest,ytest,color='g')
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)
    return ln,

def update_frame(frame):
    m,c =mc_values_anim[frame]
    x1,y1=-0.5,m*-.5+c
    x2,y2=1.5,m*1.5+c
    ln.set_data([x1,x2],[y1,y2])
    return ln,

anim=FuncAnimation(fig,update_frame,frames=range(len(mc_values_anim)),
                             init_func=init,blit=True)

HTML(anim.to_html5_video())                                           
    


# In[80]:


#PLOTTING THE REGRESSION LINE ALONG THE TRAINING DATA SET
plt.scatter(xtrain,ytrain,color='b')
plt.plot(xtrain,(m*xtrain+c),color='r')


# In[82]:


#PLOTTING ERROR VALUES
plt.plot(np.arange(len(error_values)),error_values)
plt.ylabel('Error')
plt.xlabel('Iterations')


# In[ ]:


#error values are decreasing with each iteration and the plot is parallel after a certain number of iterations


# #PREDICTION-USING THE VALUES OBTAINED FOR m and c from previous step,we obtain predictions for the values in the testing data

# In[83]:


#calculate the predictions on the test set as a vectorized operation
predicted=(m*xtest)+c


# In[84]:


#Compute MSE for the predicted values on the testing set
mean_squared_error(ytest,predicted)


# In[87]:


#Put the xtest,ytest and predicted values into a single Dataframe so that we can see the predicted values alongside the testing set
p=pd.DataFrame(list(zip(xtest,ytest,predicted)),columns=['x','target_y','predicted_y'])
p.head()


# plot the predicted values against target values

# In[88]:


plt.scatter(xtest,ytest,color='b')
plt.plot(xtest,predicted,color='r')


# REVERT NORMALIZATION TO OBTAIN THE PREDICTED PRICE OF THE HOUSES IN $1000

# In[90]:


#Reshape to change the shape that is required by the scaler
predicted=np.array(predicted).reshape(-1,1)


# In[91]:


#Reshape to change the shape to the shape required by the scaler(removing the extra dimension by slicing the data)
#-1 represents the last element of each row
predicted=predicted.reshape(-1,1)
xtest=xtest.reshape(-1,1)
ytest=ytest.reshape(-1,1)

#To obtain the data in the orginal scale,use the function 'inverse transform'
xtest_scaled=x_scaler.inverse_transform(xtest)
ytest_scaled=y_scaler.inverse_transform(ytest)
predicted_scaled=y_scaler.inverse_transform(predicted)

#This is to remove the extra dimension
xtest_scaled=xtest_scaled[:,-1]
ytest_scaled=ytest_scaled[:,-1]
predicted_test_scaled=predicted_scaled[:,-1]

p=pd.DataFrame(list(zip(xtest_scaled,ytest_scaled,predicted_scaled)),columns=['x','target_y','predicted_y'])
p=p.round(decimals=2)
p.head()


# In[ ]:




