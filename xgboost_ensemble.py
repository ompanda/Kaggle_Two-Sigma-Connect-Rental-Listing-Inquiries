# import what we need

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier

df = pd.read_json(open("./data/train.json", "r"))

df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day


num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]
X.head()

# random state for reproducing same results
random_state = 54321

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state = 54321)


#first  algo
print('RandomForestClassifier 1')
rf1 = RandomForestClassifier(n_estimators=250, criterion='entropy',  n_jobs = -1,  random_state=random_state)
rf1.fit(X_train, y_train)
y_val_pred = rf1.predict_proba(X_val)
log_loss(y_val, y_val_pred)

print('log loss')
print(log_loss(y_val, y_val_pred))

#second algo
print('RandomForestClassifier 2')
rf2= RandomForestClassifier(n_estimators=250, criterion='gini',  n_jobs = -1, random_state=random_state)
rf2.fit(X_train, y_train)
y_val_pred = rf2.predict_proba(X_val)
log_loss(y_val, y_val_pred)

print('log loss')
print(log_loss(y_val, y_val_pred))

#3rd algo- gradient boosting
print('GradientBoostingClassifier')
gbc = GradientBoostingClassifier(random_state=random_state)
gbc.fit(X_train, y_train)
y_val_pred = gbc.predict_proba(X_val)
log_loss(y_val, y_val_pred)

print('log loss')
print(log_loss(y_val, y_val_pred))

#xgboost
print('xgboost')
xgb = XGBClassifier(seed=random_state)
xgb.fit(X_train, y_train)
y_val_pred = xgb.predict_proba(X_val)
log_loss(y_val, y_val_pred)

print('log loss')
print(log_loss(y_val, y_val_pred))

#ensemble using voting algorithm
df_test = pd.read_json(open("./data/test.json", "r"))

df_test["num_photos"] = df_test["photos"].apply(len)
df_test["num_features"] = df_test["features"].apply(len)
df_test["num_description_words"] = df_test["description"].apply(lambda x: len(x.split(" ")))
df_test["created"] = pd.to_datetime(df_test["created"])
df_test["created_year"] = df_test["created"].dt.year
df_test["created_month"] = df_test["created"].dt.month
df_test["created_day"] = df_test["created"].dt.day

X_test = df_test[num_feats]


print('VotingClassifier ensemble')
eclf = VotingClassifier(estimators=[
    ('rf1', rf1), ('rf2', rf2), ('gbc', gbc), ('xgb',xgb)], voting='soft')
eclf.fit(X_train, y_train)
y_val_pred = eclf.predict_proba(X_test)

print('y predicted')
print(y_val_pred)

print('log loss')
#print(log_loss(y_val, y_val_pred))


#ensemble using voting algorithm with weights
'''eclf = VotingClassifier(estimators=[
    ('rf1', rf1), ('rf2', rf2), ('gbc', gbc), ('xgb',xgb)], voting='soft', weights = [3,1,1,1])
eclf.fit(X_train, y_train)
y_val_pred = eclf.predict_proba(X_val)
log_loss(y_val, y_val_pred)
'''

print(eclf.classes_)

#out_df = pd.DataFrame(y_val_pred)
#out_df.columns =['high',  'medium','low']

#out_df.columns = lbl.inverse_transform(out_df.columns)
#out_df["listing_id"] = df_test["listing_id"]

#out_df.to_csv("./output/submission.csv", index=False)

interest_level = {label: i for i, label in enumerate(eclf.classes_)}

ouput_submission = pd.DataFrame()
ouput_submission["listing_id"] = df_test["listing_id"]
for label in ["high", "medium", "low"]:
    ouput_submission[label] = y_val_pred[:, interest_level[label]]
ouput_submission.to_csv("./output/submission.csv", index=False)

print('Done')