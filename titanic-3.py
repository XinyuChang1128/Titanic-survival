# -*- coding: utf-8 -*-
"""
Created on Fri Feb 09 22:39:57 2018

@author: Xinyu Chang
"""
import numpy as np
import pandas as pd
import xgboost as xgb
train = pd.read_csv("C://Users/Xinyu Chang/PycharmProjects/titanic/train.csv", dtype={"Age": np.float64},)
test = pd.read_csv("C://Users/Xinyu Chang/PycharmProjects/titanic/test.csv", dtype={"Age": np.float64},)
train.describe()

x=[train[(train.Sex=='male')]['Sex'].size,train[(train.Sex=='female')]['Sex'].size]
y=[train[(train.Sex=='male') & (train.Survived == 1)]['Sex'].size,train[(train.Sex=='female') & (train.Survived == 1)]['Sex'].size]
print 'male number:'+str(x[0])+'    '+'female number:'+str(x[1])

print 'male survive:'+str(y[0])+'    '+'female survive:'+str(y[1])

#using average number
train.Embarked[train.Embarked.isnull()] = train.Embarked.dropna().mode().values

train.Cabin[train.Cabin.isnull()]='U0'

def harmonize_data(titanic):

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0.0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1.0

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0.0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1.0
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2.0

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

def create_submission(alg, train, test, predictors, filename):

    alg.fit(train[predictors], train["Survived"])
    predictions = alg.predict(test[predictors])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })

    submission.to_csv(filename, index=False)

train_data = harmonize_data(train)
test_data  = harmonize_data(test)

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg    = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(
    alg,
    train_data[predictors],
    train_data["Survived"],
    cv=3
)

print(scores.mean())

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


alg = RandomForestClassifier(
    random_state=1,
    n_estimators=50,
    min_samples_split=4,
    min_samples_leaf=2,
    max_depth=10
)

# specify parameters via map


#xgboost
#alg = xgb.XGBClassifier(
#    max_depth=3, 
#    n_estimators=300,
#    learning_rate=0.05
#    )



#AdaBoost
#alg =AdaBoostClassifier(
 #   n_estimators=50,
#    learning_rate=1
#  )


scores = cross_validation.cross_val_score(
    alg,
    train_data[predictors],
    train_data["Survived"],
)

print(scores.mean())

create_submission(alg, train_data, test_data, predictors, "C://Users/Xinyu Chang/PycharmProjects/titanic/run-04.csv")
