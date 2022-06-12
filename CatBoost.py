import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

balanced_X_train = pd.read_csv("balanced_X_train.csv")
balanced_X_test = pd.read_csv("balanced_X_test.csv")
balanced_y_train = pd.read_csv("balanced_y_train.csv")
balanced_y_test = pd.read_csv("balanced_y_test.csv")

cbc = CatBoostClassifier()

cbc.fit(balanced_X_train, balanced_y_train, early_stopping_rounds = 10)

print(f'CatBoostClassifier train : {roc_auc_score(balanced_y_train, cbc.predict_proba(balanced_X_train)[:, 1])}')
print(f'CatBoostClassifier test: {roc_auc_score(balanced_y_test, cbc.predict_proba(balanced_X_test)[:, 1])}')