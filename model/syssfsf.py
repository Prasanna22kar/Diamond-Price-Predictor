

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#import seaborn as sns
import pickle 
import joblib


# In[2]:


data = pd.read_csv("E:\pj 1\diamonds.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data = data.drop(columns = 'Unnamed: 0')


# In[6]:


data.head()


# In[7]:


data.nunique()


# In[8]:


data[data.duplicated()]


# In[9]:


data= data.drop_duplicates()


# In[10]:


data[data.duplicated()]


# In[11]:


from sklearn.preprocessing import LabelEncoder


# In[12]:


lnc = LabelEncoder()
df = data.copy()
for i in data:
    r=data[i].dtypes
    if r=='object':
        df[i]=lnc.fit_transform(data[i])
        joblib.dump(lnc,i+'.joblib',compress=9)


# In[13]:


df.head


# In[14]:


df.info()


# In[15]:


for i in df :
    r=df[i].dtypes
    if r == 'float64':
        plt.scatter(range(len(df[i])),df[i], label=i)
        plt.legend()
        plt.show()


# In[16]:


df= df[(df.meas_length < 40)]
df= df[(df.meas_width < 30)]
df= df[(df.meas_depth < 30)]


# In[17]:


df[(df.meas_length == 0) | (df.meas_width == 0) | (df.meas_depth == 0) | (df.depth_percent == 0) | (df.table_percent == 0)]


# In[18]:


df2= df.copy()
df2= df2[(df2.meas_length != 0)]
df2= df2[(df2.meas_width != 0)]
df2= df2[(df2.meas_depth != 0)]
df2= df2[(df2.depth_percent != 0)]
df2= df2[(df2.table_percent != 0)]

print(df.shape)
print(df2.shape)


# In[19]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss=MinMaxScaler()

corr= df2.corr()
#sns.heatmap(corr)


# In[20]:


features =[]
for i in df2:
    if i != 'total_sales_price':
        features.append(i)
print(features)


# In[21]:


x=df2[features]
y=df2['total_sales_price']

print(x.shape)
print(y.shape)


# In[22]:


from sklearn.feature_selection import SelectKBest as skb
from sklearn.feature_selection import f_regression

fs= skb(f_regression,k='all')
fs.fit(x,y)
for i in range(len(fs.scores_)):
    print('Feature %d: %f' %(i,fs.scores_[i]))


# In[23]:


fs = skb(f_regression,k=15)
xnew=fs.fit_transform(x,y)
f= np.array(features)
flit=fs.get_support()
f=f[flit]
print(f,len(f))


# In[24]:


xnew=x
y=y.values.reshape(-1,1)
print(xnew.shape)
print(y.shape)


# In[25]:


from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test = tts(xnew,y, test_size=0.2, random_state = 40)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[26]:


from sklearn.linear_model import LinearRegression

mlr= LinearRegression()

mlr.fit(x_train,y_train)


# In[27]:


y_pred_mlr = mlr.predict(x_test)


# In[28]:


from sklearn import metrics
MABE = metrics.mean_absolute_error(y_test, y_pred_mlr)
r2 = mlr.score(x_test,y_test)*100


# In[29]:


print(r2,MABE)


# In[31]:


import tensorflow
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


# In[32]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense




# In[33]:


model = Sequential()
model.add(Dense(units=100 , input_dim=24, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=50, kernel_initializer='normal', activation='relu'))
model.add(Dense(1,kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')
history= model.fit(x_train,y_train,validation_split=0.2, batch_size=2000, epochs=50, verbose=1)


# In[34]:


pd.DataFrame(history.history).plot(figsize=(10,10))
plt.show()


# In[35]:


pred = model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test,pred)*100

print(r2)


# In[36]:


model.save('E:/pj 1/app/model/tf_m_1.0.0.h5')


# In[ ]:




