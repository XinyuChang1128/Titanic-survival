import numpy as np
import pandas as pd

train = pd.read_csv("C://Users/Xinyu Chang/Documents/2018Titanic/train.csv", dtype={"Age": np.float64},)
test = pd.read_csv("C://Users/Xinyu Chang/Documents/2018Titanic/test.csv", dtype={"Age": np.float64},)
train.describe()

def harmonize_data(titanic):

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

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
    cv=10
)

print(scores.mean())

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]


alg = RandomForestClassifier(
    random_state=1,
    n_estimators=150,
    min_samples_split=4,
    min_samples_leaf=2,
    max_depth=20
)


# AdaBoost
# alg =AdaBoostClassifier(
#     random_state=42,
#     n_estimators=500,
#     learning_rate={0.5, 0.6},
# )


scores = cross_validation.cross_val_score(
    alg,
    train_data[predictors],
    train_data["Survived"],
    cv=3
)

print(scores.mean())

create_submission(alg, train_data, test_data, predictors, "C://Users/Xinyu Chang/Documents/2018Titanic/run-02.csv")