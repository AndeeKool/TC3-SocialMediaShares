from sklearn import neighbors, linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("OnlineNewsPopularity.csv", sep=", ")

#X will be a DataFrame
# y will be the label

w = df.iloc[["weekday_is_monday", " weekday_is_tuesday", "weekday_is_wednesday", "weekday_is_thursday", "weekday_is_friday", "weekday_is_saturday", "weekday_is_sunday"]]
y = df[["shares"]]

# Preproccessing
# le = preprocessing.LabelEncoder()

days = w.mul(range(7), fill_value=0)
days = days.agg("sum", axis="columns")
df[["days"]] = days


# X.loc[:, ("weekday_is_monday")] = le.fit_transform(X[["weekday_is_monday"]])
# X.loc[:, ("weekday_is_tuesday")] = le.fit_transform(X[["weekday_is_tuesday"]])
# X.loc[:, ("weekday_is_wednesday")] = le.fit_transform(X[["weekday_is_wednesday"]])
# X.loc[:, ("weekday_is_thursday")] = le.fit_transform(X[["weekday_is_thursday"]])
# X.loc[:, ("weekday_is_friday")] = le.fit_transform(X[["weekday_is_friday"]])
# X.loc[:, ("is_weekend")] = le.fit_transform(X[["is_weekend"]])

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=93)

model = neighbors.KNeighborsRegressor(n_neighbors=5)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Accuracy
acc = model.score(X_test, y_test)

print(f"Accuracy: {acc}")

