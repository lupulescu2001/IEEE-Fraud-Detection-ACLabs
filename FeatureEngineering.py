import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import re

df1 = pd.read_csv("train_transaction.csv")
df2 = pd.read_csv("train_identity.csv")
df = df1.merge(df2, on = 'TransactionID', how = 'left')

keep_columns = df.isnull().sum() / df.shape[0] * 100 < 90
df = df.loc[:, keep_columns]

df['id_34'] = df['id_34'].replace({"match_status:-1" : "match_status:1"})

corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
df.drop(to_drop, axis=1, inplace=True)

def drop_string_columns(df):
    column_array = np.array([])
    for column in df.columns:
        if (df[column].dtypes == 'object' and df[column].unique().size >= 7):
            column_array = np.append(column_array, column)
    return column_array

df.drop(columns = drop_string_columns(df), inplace = True)

def get_string_columns(df):
    column_array = np.array([])
    for column in df.columns:
        if (df[column].dtypes == 'object'):
            column_array = np.append(column_array, column)
    return column_array

string_columns = get_string_columns(df)

def fill_na_string_columns(string_columns):
    for column in string_columns :
        df[column].fillna("does not exist", inplace = True)
fill_na_string_columns(string_columns)

df.fillna(0, inplace = True)
X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(df.drop(columns = ['isFraud'])), df['isFraud'], test_size=0.33, random_state=42)

X_train_for_balance = X_train[y_train == 1]
balanced_X_train = pd.concat([X_train, pd.concat([X_train_for_balance] * 25, ignore_index = True)], ignore_index = True)
y_train_for_balance = y_train[y_train == 1]
balanced_y_train = pd.concat([y_train, pd.concat([y_train_for_balance] * 25, ignore_index = True)], ignore_index = True)

X_test_for_balance = X_test[y_test == 1]
balanced_X_test = pd.concat([X_test, pd.concat([X_test_for_balance] * 25, ignore_index = True)], ignore_index = True)
y_test_for_balance = y_test[y_test == 1]
balanced_y_test = pd.concat([y_test, pd.concat([y_test_for_balance] * 25, ignore_index = True)], ignore_index = True)

idx_train = np.random.permutation(balanced_y_train.index)
idx_test = np.random.permutation(balanced_y_test.index)
balanced_y_test.reindex(idx_test)
balanced_X_test.reindex(idx_test)
balanced_y_train.reindex(idx_train)
balanced_X_train.reindex(idx_train)

scaler = MinMaxScaler()
balanced_X_train = pd.DataFrame(scaler.fit_transform(balanced_X_train), index=balanced_X_train.index, columns=balanced_X_train.columns)
balanced_X_test = pd.DataFrame(scaler.fit_transform(balanced_X_test), index=balanced_X_test.index, columns=balanced_X_test.columns)

balanced_X_train = balanced_X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
balanced_X_test = balanced_X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

balanced_X_train.to_csv("balanced_X_train.csv", index = False)
balanced_X_test.to_csv("balanced_X_test.csv", index = False)
balanced_y_train.to_csv("balanced_y_train.csv", index = False)
balanced_y_test.to_csv("balanced_y_test.csv", index = False)

