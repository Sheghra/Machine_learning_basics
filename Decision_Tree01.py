import pandas as pd #pandas will be used to read the csv data and manipulate them
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#used to read the csv file
df=pd.read_csv('data1.csv')

#this part of the code is used to convert the non-integer to integer values
d={'UK':0,'USA':1,'N':2}
df['Nationality']=df['Nationality'].map(d)
d={'NO':0,'YES':1}
df['Go']=df['Go'].map(d)

#separate the X=features and Y=target columns 
#basically feautres list will columns using which we will make the predictions and target column will
#contain the values which we want to predict
feature=['Age','Experience','Rank','Nationality']
x=df[feature]
y=df['Go']

dtree=DecisionTreeClassifier() #this creates the tree object which
dtree=dtree.fit(x,y)

tree.plot_tree(dtree,feature_names=feature)
plt.show()
