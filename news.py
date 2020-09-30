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

#-----------#


c = df[["data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world"]]
channels = c.mul(range(6), fill_value=0)
channels = channels.agg("sum", axis="columns")
df[["channels"]] = channels


# X = df[["avg_positive_polarity", "rate_positive_words", "global_subjectivity", "LDA_02", "num_imgs"]]
X = df[["avg_positive_polarity", "rate_positive_words", "global_subjectivity", "LDA_02", "num_imgs", "num_hrefs"]]
y = df[["shares"]]

# Subplots de pyplot
column_names = ["avg_positive_polarity", "rate_positive_words", "global_subjectivity", "LDA_02", "num_imgs", "num_hrefs"]

# 6 datos -> 2 (rows) x 3 (columns)
fig, axs = plt.subplots(2, 3)
for index, ax in enumerate(axs.flat):
    ax.scatter(df[[column_names[index]]], y, y=0.5)
    ax.set_title(column_names[index])
    ax.set(ylabel="scores")

plt.show()


# X[["num_hrefs"]] = scaler.fit_transform(X[["num_hrefs"]])
# X[["num_imgs"]] = scaler.fit_transform(X[["num_imgs"]])
# y = normalizer.fit_transform(y[["shares"]])

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, random_state=23)

# model = neighbors.KNeighborsRegressor(n_neighbors=5, weights="uniform")
# model = linear_model.LinearRegression()

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# acc = model.score(X_test, y_test)

# print(f"Accuracy: {acc}")


# plt.plot(X_test, y_pred)

# plt.scatter(X_test[["avg_positive_polarity"]], y_pred)
# plt.scatter(X_test[["rate_positive_words"]], y_pred)
# plt.scatter(X_test[["global_subjectivity"]], y_pred)

# plt.show()