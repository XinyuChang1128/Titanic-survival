import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string as str
import sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import csv
from sklearn.ensemble import ExtraTreesClassifier
#Read csv file
train_data=pd.read_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/train.csv')
test_data=pd.read_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/test.csv')

# info
#train_data.info()
#test_data.info()

#show pie

#0 means survived  1 means not survived

#train_data['Survived'].value_counts().plot.pie(autopct='%1.1f%%')
train_data.groupby(['Sex','Survived'])['Survived'].count()
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()

train_data.groupby(['Sex','Pclass','Survived'])['Survived'].count()
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(color=['r','g','b'])
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
train_data.groupby(['Sex','Pclass','Survived'])['Survived'].count()

f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train_data,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train_data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
#plt.show()

# before this line is working



#train and test together
train_data_org = pd.read_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/train.csv')
test_data_org = pd.read_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/test.csv')
test_data_org['Survived'] = 0
combined_train_test = train_data_org.append(test_data_org)
# about the embank missing data
if combined_train_test['Embarked'].isnull().sum() != 0:
    combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'],prefix=combined_train_test[['Embarked']].columns[0])
    combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
# sex seperate
sex_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, sex_dummies_df], axis=1)

combined_train_test['Title'] = combined_train_test['Name'].str.extract('.+,(.+)').str.extract( '^(.+?)\.').str.strip()
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master'], 'Master'))
combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
title_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, title_dummies_df], axis=1)

if combined_train_test['Fare'].isnull().sum() != 0:
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform('mean'))

combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)


train_data = combined_train_test[:891]
test_data = combined_train_test[891:]
titanic_train_data_X = train_data.drop(['Survived'], axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'], axis=1)


def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # random forest
    rf_est = RandomForestClassifier(
        random_state=1,
        n_estimators=150,
        min_samples_split=4,
        min_samples_leaf=2,
        max_depth=20

    )
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    # feature accoding to the Importance sort
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),'importance': rf_grid.best_estimator_.feature_importances_}).sort_values( 'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print(str(features_top_n_rf[:25]))
    # collection the top three features_top_n_et
   # features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et],ignore_index=True).drop_duplicates()

    return features_top_n


    feature_to_pick = 250
    titanic_test_data_X['Survived'] = 0
submission = pd.DataFrame({'PassengerId':test_data_org.loc[:,'PassengerId'],'Survived':titanic_test_data_X.loc[:,'Survived']})
    #submission.to_csv('C://Users/Xinyu Chang/PycharmProjects/titanic/submission_result.csv', index=False, sep=',')
submission.to_csv(rf_est, train_data, test_data, predictors, "C://Users/Xinyu Chang/PycharmProjects/titanic/submission_result.csv")