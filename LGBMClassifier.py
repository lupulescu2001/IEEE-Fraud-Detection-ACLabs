from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score

balanced_X_train = pd.read_csv("balanced_X_train.csv")
balanced_X_test = pd.read_csv("balanced_X_test.csv")
balanced_y_train = pd.read_csv("balanced_y_train.csv")
balanced_y_test = pd.read_csv("balanced_y_test.csv")

lgbm = LGBMClassifier()
lgbm.fit(balanced_X_train, balanced_y_train)

print(f'LGBMClassifier train : {roc_auc_score(balanced_y_train, lgbm.predict_proba(balanced_X_train)[:, 1])}')
print(f'LGBMClassifier test : {roc_auc_score(balanced_y_test, lgbm.predict_proba(balanced_X_test)[:, 1])}')