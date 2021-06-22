#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install nomkl


# In[1]:


conda install seaborn


# In[101]:


import pandas as pd
import numpy as np


# In[102]:


from sklearn.linear_model import LogisticRegression

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# In[103]:



import seaborn as sns


# In[104]:



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[105]:


from sklearn.linear_model import LogisticRegression


# In[106]:


import joblib


# In[107]:


import pandas as pd
diabetesDF = pd.read_csv('diabetes.csv')


# In[ ]:





# In[108]:


get_ipython().run_cell_magic('HTML', '', '<style type="text/css">\n\n    table td,th {\n        broder: 1px black solid !important;\n        colour: black !important;\n}\n</style>')


# In[109]:


diabetesDF.head()


# # missing and zero values

# In[ ]:





# In[110]:


diabetesDF.info()


# In[111]:


diabetesDF.describe()


# In[114]:


print(diabetesDF.isnull().any().sum())
print(diabetesDF.isnull().sum())


# In[116]:


diabetesDF=diabetesDF.drop("Insulin",axis=1)


# In[117]:


diabetesDF=diabetesDF.dropna()
diabetesDF.shape


# In[118]:


diabetesDF.groupby('Outcome').mean()


# In[ ]:





# In[120]:


## to find where none of teh columns has 0 value(except the first and last column)
diabetesDF=diabetesDF[-(diabetesDF[diabetesDF.columns[1:-1]]==0).any(axis=1)]
diabetesDF.shape


# In[121]:


diabetesDF.groupby('Outcome').agg(['mean','median'])


# In[ ]:





# # Histogram plots

# In[122]:


for i,col in enumerate(diabetesDF[:-1]):
    plt.figure(i)
    sns.displot(diabetesDF[col]);


# ## scatter matrix
# 

# In[123]:


sns.pairplot(diabetesDF, hue="Outcome", diag_kind="hist");


# ## correlation plots

# In[17]:


diabetesDF.corr()


# In[18]:


plt.figure(figsize=(9,9))
sns.heatmap(np.abs(diabetesDF.corr()), annot=True, cmap="viridis", fmt="0.2f");


# ## box plots, violin plots and bee swarm plots
# 

# In[19]:


sns.boxplot(x="Outcome", y="BMI", data=diabetesDF, whis=3.0);
sns.swarmplot(x="Outcome", y="BMI", data=diabetesDF, size=2, color="k", alpha=0.3);


# In[20]:


sns.violinplot(x="Outcome", y="BMI", data=diabetesDF);
sns.swarmplot(x="Outcome", y="BMI", data=diabetesDF, size=2, color="k", alpha=0.3);


# ## 2D Histograms
# useful when you have a lot of data..i.e. at least 1000's of points

# In[21]:


plt.figure(figsize=(10,9))
plt.hist2d(diabetesDF["Glucose"],diabetesDF["BMI"], bins=(20,20),cmap="magma")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.colorbar();


# In[22]:


## remember we do not have a large number of points
hist,x_edge,y_edge = np.histogram2d(diabetesDF["Glucose"],diabetesDF["BMI"], bins=20)
#here we want the centers of the bins
x_center=0.5 * (x_edge[1:] + x_edge[:-1])
y_center=0.5 * (y_edge[1:] + y_edge[:-1])
plt.contour(x_center, y_center, hist, levels=6)
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.colorbar();


# In[23]:


plt.figure(figsize=(9,9))
sns.kdeplot(diabetesDF["Glucose"],diabetesDF["BMI"], cmap="viridis", bw_methods=(4,4));
plt.hist2d(diabetesDF["Glucose"], diabetesDF["BMI"], bins=20, cmap="magma", alpha=0.5) 
plt.colorbar();


# In[24]:


plt.figure(figsize=(9,9))
sns.kdeplot(diabetesDF["Glucose"], diabetesDF["BMI"],cmap="magma",shade=True,cbar=True);


# In[25]:


plt.figure(figsize=(9,9))
m=diabetesDF["Outcome"]==1
plt.scatter(diabetesDF.loc[m, "Glucose"],diabetesDF.loc[m,"BMI"], c="r", s=15, label="Diabetic")
plt.scatter(diabetesDF.loc[-m, "Glucose"],diabetesDF.loc[-m,"BMI"], c="b", s=15, label="Non-Diabetic")
plt.xlabel("Glucose")
plt.ylabel("BMI")
plt.legend(loc=2);


# In[26]:


pip install chainconsumer


# In[27]:


params=["Glucose","BMI"]
m=diabetesDF["Outcome"]==1
diabetic=diabetesDF.loc[m,params].values
non_diabetic=diabetesDF.loc[-m,params].values
non_diabetic.shape


# In[28]:


from chainconsumer import ChainConsumer
c= ChainConsumer()
c.add_chain(diabetic,parameters=params, name="Diabetic", kde=1.0, color="b")
c.add_chain(non_diabetic, parameters=params, name="Non Diabetic",kde=1.0, color="r")
c.configure(contour_labels="confidence",usetex=False)
c.plotter.plot(figsize=3.0);


# * It gives you two contours for each chain, one of the 68% confidence interval and the other is for the 95% confidence interval.
# 
# * This is useful when doing Hypothesis testing.
# 
# * For example, if you randomly pick a Diabetic person out of a different data sample, you can say that 68% of the time would expect glucose levels to lie within the 68% contour, and 95% of the time their BMI and Glucose levels would lie in the second contour.
# 
# * This is useful when you would like to check if a data point comes from this distribution or not.
# 
# * For example, you can check where the data point lies and estimate the  chance of it being of a diabetic or Non Diabetic person.
# 

# In[29]:


c.plotter.plot_summary(figsize=2.0);


# * if we ignore the correlations, we ca visually see that Diabetic individuals have higher Glucose levels that Non Diabetic individuals.
# * They also tend to have a higher BMI(look at hte tall of the curve)
# 

# # A probabilistic Analysis

# In[30]:


diabetesDF=diabetesDF[["Glucose","BMI","Age", "Outcome"]]


# In[31]:


diabetesDF.shape


# In[32]:


import plotly.graph_objects as go
df_y= diabetesDF.loc[diabetesDF["Outcome"]==1,["Glucose","BMI", "Age"]]
df_n= diabetesDF.loc[diabetesDF["Outcome"]==0, ["Glucose","BMI", "Age"]]
## crete an interactive 3D plot using the two subdatasets
fig=go.Figure()
fig.add_trace(go.Scatter3d(x=df_y["Glucose"],y=df_y["BMI"],  z=df_y["Age"], mode="markers", name="Diabetic"))
fig.add_trace(go.Scatter3d(x=df_n["Glucose"],y=df_y["BMI"],  z=df_y["Age"], mode="markers", name="Non Diabetic"))
fig.show()


# In[33]:


sns.set(style="ticks")
sns.pairplot(df_y,diag_kind="hist");


# In[34]:


sns.set(style="ticks")
sns.pairplot(df_n,diag_kind="hist");


# In[35]:


df_y.shape


# In[36]:


df_n.shape


# In[37]:


test_point=[110,35,52]

fig=go.Figure()
fig.add_trace(go.Scatter3d(x=df_y["Glucose"],y=df_y["BMI"],z=df_y["Age"],mode="markers",name="Diabetic"))
fig.add_trace(go.Scatter3d(x=df_n["Glucose"],y=df_n["BMI"],z=df_n["Age"],mode="markers",name="Non Diabetic"))
fig.add_trace(go.Scatter3d(x=[test_point[0]],y=[test_point[1]],z=[test_point[2]],mode="markers",name="Test"))


# In[38]:


from scipy.stats import multivariate_normal as mn
prob_test=[]
for d in [df_y,df_n]:
    mean=np.mean(d)
    cov=np.cov(d,rowvar=0)
    prob=mn.pdf(test_point,mean,cov)
    prob_test.append(prob)


# In[39]:


prob_test


# In[40]:


num_y=df_y.shape[0]
num_n=df_n.shape[0]
print("Number of people with Diabetes is: ", num_y)
print("Number of people without Diabetes is: ", num_n)
prob_diagnosis=num_n * prob_test[0]/(num_y * prob_test[0] + num_n *prob_test[1])
print(f"negative diagnosis chance is { 100 * prob_diagnosis:.2f}%")


# # predicting Diabetes with logistic Regression
# 

# In[124]:


diabetesDF.head()


# In[125]:


X,y= diabetesDF.values[:,:-1],diabetesDF.values[:,-1]


# In[126]:


from sklearn.model_selection import train_test_split


# In[130]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01,random_state=42,stratify=y)


# In[131]:


model=LogisticRegression()
model.fit(X_train,y_train)


# In[132]:


accuracy=model.score(X_test,y_test)
print("accuracy", accuracy * 100, "%")


# In[135]:


coeff = list(diabetesCheck.coef_[0])
labels = list(dfTrain.drop('Outcome',1).columns)


# In[136]:


import pandas as pd
features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# In[ ]:




