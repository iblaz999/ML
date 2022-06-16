import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix

from sklearn import preprocessing





data = pd.read_csv("C:\\Users\\iblazevic1\\Desktop\\DATA_ML_ZAD\\athlete_events.csv")

data['Medal'] = data['Medal'].apply(lambda x: 1 if str(x) != 'nan' else 0)


model = preprocessing.LabelEncoder()


ID = model.fit_transform(list(data["ID"]))
Name = model.fit_transform(list(data["Name"]))
Sex = model.fit_transform(list(data["Sex"]))
Age = model.fit_transform(list(data["Age"]))
Height = model.fit_transform(list(data["Height"]))
Weight = model.fit_transform(list(data["Weight"]))
Team = model.fit_transform(list(data["Team"]))
NOC = model.fit_transform(list(data["NOC"]))
Games = model.fit_transform(list(data["Games"]))
Year = model.fit_transform(list(data["Year"]))
Season = model.fit_transform(list(data["Season"]))
City = model.fit_transform(list(data["City"]))
Sport = model.fit_transform(list(data["Sport"]))
Event = model.fit_transform(list(data["Event"]))
Medal = model.fit_transform(list(data["Medal"]))






predict="Medal"



X=list(zip(ID,Name,Sex,Age,Height,Weight,Team,NOC,Games,Year,Season,City,Sport,Event))
y=list(Medal)

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
model=KNeighborsClassifier(n_neighbors=9)
model.fit(x_train,y_train)
acc = model.score(x_test,y_test)
print(acc)


predicted = model.predict(x_test)
print(predicted)
names=[0,1]


# for x   in range(len(predicted)):
#     print("Predicted:",predicted[x],"Data:",x_test[x],"Actual:",y_test[x])