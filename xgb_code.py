import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

df_train = pd.read_csv("encoded_train.csv",  encoding='ISO-8859-1') 
df_test = pd.read_csv("encoded_submission.csv",  encoding='ISO-8859-1') 

x_train, x_val, y_train, y_val = train_test_split(
    df_train.drop("is_converted", axis=1),
    df_train["is_converted"],
    test_size=0.2,
    shuffle=True,
    random_state=400
)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(x_train, y_train)

def get_clf_eval(y_test, y_pred=None):
    confusion = confusion_matrix(y_test, y_pred, labels=[True, False])
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, labels=[True, False])
    recall = recall_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, labels=[True, False])

    print("오차행렬:\n", confusion)
    print("\n정확도: {:.4f}".format(accuracy))
    print("정밀도: {:.4f}".format(precision))
    print("재현율: {:.4f}".format(recall))
    print("F1: {:.4f}".format(F1))

pred = model.predict(x_val.fillna(0))
get_clf_eval(y_val, pred)

x_test = df_test.drop(["is_converted"], axis=1)
test_pred = model.predict(x_test.fillna(0))
sum(test_pred) 

df_sub = pd.read_csv("encoded_submission.csv")
df_sub["is_converted"] = test_pred

df_sub.to_csv("submission.csv", index=False)