import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn import preprocessing




data = pd.read_csv("C:\\Users\\iblazevic1\\Desktop\\DATA_ML_ZAD\\athlete_events.csv")

data['Medal'] = data['Medal'].apply(lambda x: 1 if str(x) != 'nan' else 0)

predict = 'Medal'

X = np.array(data['ID'])

y = np.array(data[predict])

X.reshape(-1, 1)

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




x_train,y_train,x_test,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.7)

linear=linear_model.LinearRegression()

x_train = np.array(x_train)
x_train.reshape(-1, 1)

y_train = np.array(y_train)
y_train.reshape(-1, 1)


linear.fit(x_train, y_train)
acc=linear.score (x_test, y_test)
print (acc)