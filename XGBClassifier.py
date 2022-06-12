import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

balanced_X_train = pd.read_csv("balanced_X_train.csv")
balanced_X_test = pd.read_csv("balanced_X_test.csv")
balanced_y_train = pd.read_csv("balanced_y_train.csv")
balanced_y_test = pd.read_csv("balanced_y_test.csv")

xgb = XGBClassifier()
xgb.fit(balanced_X_train, balanced_y_train)
print(f'XGBClassifier train : {roc_auc_score(balanced_y_train, xgb.predict_proba(balanced_X_train)[:, 1])}')
print(f'XGBClassifier test : {roc_auc_score(balanced_y_test, xgb.predict_proba(balanced_X_test)[:, 1])}')