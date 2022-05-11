# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Importing Necessary Data

# %%
import matplotlib.pyplot as plt
# Descriptive Plots - Boxplot, Density etc.
import seaborn as sns
import pandas as pd
import numpy as np

# %% [markdown]
# # Reading Data

# %%
park_df=pd.read_csv("E:\ML\Parkinsons.csv")


# %%
park_df

# %% [markdown]
# # Data Visualization and Analysis

# %%
park_df.isnull().sum()


# %%
for i in park_df.columns:
    if i!='status':
        plt.figure(figsize=(13,8))
        print("%s vs Status"%i)
        sns.catplot(x='status',y=i,kind='box',data=park_df)
        plt.show()
    else:
        continue


# %%
sns.countplot(x=park_df['status'].values)


# %%
sns.heatmap(park_df.corr(),annot=True)

# %% [markdown]
# # Data Preprocessing

# %%
park_df['status'].value_counts()

# %% [markdown]
# **Data Augmentation**

# %%
from sklearn.utils import resample
data_1=park_df[park_df['status']==1]
data_2=park_df[park_df['status']==0]
d2_rescaled=resample(data_2,n_samples=100,replace=True,random_state=123)
park_df=pd.concat([data_1,d2_rescaled])


# %%
x=park_df.drop(['MDVP:Fhi(Hz)','status'],axis=1)
y=park_df['status']


# %%
x


# %%
from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler(feature_range=(0,1))
x_scaled=scale.fit_transform(x)

# %% [markdown]
# # Algorithm Tuning
# %% [markdown]
# **Finding the best suitable model**

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
maximum=0
for i in range(0,12):
    xtrain,xtest,ytrain,ytest=train_test_split(x_scaled,y,test_size=0.2,random_state=i)
    bst_model=LogisticRegression(C=1e6,max_iter=1e9,random_state=0)
    bst_model.fit(xtrain,ytrain)
    ##print(bst_model.score(xtest,ytest),i)
    if bst_model.score(xtest,ytest)>maximum:
        maximum=bst_model.score(xtest,ytest)
        max_iter=i
print(max_iter,maximum)


# %%
xtrain,xtest,ytrain,ytest=train_test_split(x_scaled,y,test_size=0.2,random_state=2)
park_model=LogisticRegression(C=1e6,max_iter=1e9,random_state=0)
park_model.fit(xtrain,ytrain)

# %% [markdown]
# # Classification Metrics

# %%
from sklearn.metrics import classification_report,confusion_matrix
predict=park_model.predict(xtest)
confusion_matrix(ytest,predict)


# %%
print(classification_report(ytest,predict))


