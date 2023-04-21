# import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

df = pd.read_csv("data/Social_Network_Ads.csv", sep=";")
# normalize dataset
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

x = df.iloc[:, :2].to_numpy()
y = df["Purchased"].to_numpy()

# Split dataset
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(x, y, test_size=0.2, random_state=19521338)

# Train model
logreg = LogisticRegression(random_state=19521338)
logreg.fit(X_train, y_train)

train_result = round(
    f1_score(y_train, logreg.predict(X_train), average="weighted"), 5
)

test_result = round(
    f1_score(y_test, logreg.predict(X_test), average="weighted"), 5
)

# get the result
result = []
data_shape = []

result.append(test_result)
result.append(train_result)

data_shape.append(X_train.shape)
data_shape.append(y_train.shape)

data_shape.append(y_test.shape)
data_shape.append(y_test.shape)
