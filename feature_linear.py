import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/50_Startups.csv")

Feature = df[["R&D Spend", "Administration", "Marketing Spend"]]
Feature = pd.concat([Feature, pd.get_dummies(df["State"])], axis=1)

X = Feature

y = df["Profit"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

print("Training data shape: ", X_train.shape, y_train.shape)
print("validation data shape: ", X_test.shape, y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Done")

y_pred = lr.predict(X_test)

lr.score(X_train, y_train)
lr.score(X_test, y_test)

mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
mse = mean_squared_error(y_true=y_test, y_pred=y_pred)  # default=True
rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
