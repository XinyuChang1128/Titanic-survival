import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string as str


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import cross_validation as cv
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/train.csv')
test_data = pd.read_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/test.csv')

#train_data.info()
#print(train_data.describe())
#print(train_data['Age'])
#把年龄填充中位数
train_data['Age']=train_data['Age'].fillna(train_data['Age'].median())
#print(train_data.describe())

#0 means male  1 means female
train_data.loc[train_data['Sex']=='male','Sex'] = 0
train_data.loc[train_data['Sex']=='female','Sex'] = 1
#print(train_data['Sex'])
#print(train_data.info())

print(train_data["Embarked"].unique())
train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.loc[train_data['Embarked'] == 'S', 'Embarked'] = 0
train_data.loc[train_data['Embarked'] == 'C', 'Embarked'] = 1
train_data.loc[train_data['Embarked'] == 'Q', 'Embarked'] = 2
print(train_data.info())

#using to the predict
predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

alg=LinearRegression()
#kf = KFold(n_splits=2)
#kf.get_n_splits(train_data)

#print(kf)  
#KFold(n_splits=2, random_state=None, shuffle=False)

kf=KFold(train_data.shape[0],
         shuffle=False,
         random_state=1)

for train,test in kf.split(train_data):
    train_predictors=(train_data[predictors].iloc[train,:])
    train_target = train_data["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(train_data[predictors].iloc[test,:])
    predictions.append(test_predictions)
predictions=[]   
    
classifier = LogisticRegression()  #有Age  
scores=cross_validation.cross_val_score(classifier,dt_learn_features,dt_learn_target,cv=5)  #交叉检验  
print scores,scores.mean() 

