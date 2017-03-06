from string import ascii_uppercase

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier

def hist_by(df, column):
    X, y = [], []
    for value in sorted(df[column].unique()):
        X.append(value)
        y.append(df[df[column] == value]['income'].mean() * 100.0)

    index = np.arange(len(X))
    width = 0.35
    plt.bar(index, y, width)
    plt.xticks(index+width/2, X, rotation=70)
    plt.yticks(np.arange(0, 100, 10))
    plt.ylim(0, 100)
    plt.xlabel(column)
    plt.ylabel('Percentage of people who\'s income is above $50K')
    plt.tight_layout()
    plt.show()

def replace_binned(df, column, bins):
    group_names = list(ascii_uppercase[:len(bins)-1])
    binned = pd.cut(df[column], bins, labels=group_names)
    df[column] = binned
    return df

adult_df = pd.read_csv('./data/adult.csv')
adult_df['income'] = np.where(adult_df['income'] == '>50K', 1, 0)

adult_df = adult_df[adult_df['occupation'] != '?']
adult_df = adult_df[adult_df['workclass'] != '?']

# Some preprocessing
education_dummies = pd.get_dummies(adult_df['education'])
marital_dummies = pd.get_dummies(adult_df['marital.status'])
relationship_dummies = pd.get_dummies(adult_df['relationship'])
sex_dummies = pd.get_dummies(adult_df['sex'])
occupation_dummies = pd.get_dummies(adult_df['occupation'])
native_dummies = pd.get_dummies(adult_df['native.country'])
race_dummies = pd.get_dummies(adult_df['race'])
workclass_dummies = pd.get_dummies(adult_df['workclass'])

replace_binned(adult_df, 'capital.loss', np.arange(-1, 4500, 500))
loss_dummies = pd.get_dummies(adult_df['capital.loss'])

replace_binned(adult_df, 'capital.gain', list(range(-1, 42000, 5000)) + [100000])
gain_dummies = pd.get_dummies(adult_df['capital.gain'])

X = pd.concat([adult_df[['age', 'hours.per.week']], gain_dummies, occupation_dummies, workclass_dummies, education_dummies, marital_dummies, race_dummies, sex_dummies], axis=1)

y = adult_df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
