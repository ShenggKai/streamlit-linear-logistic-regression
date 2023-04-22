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


def get_result_lg(df, selected_input, selected_output, test_sz, random_st):
    # df = pd.read_csv("data/Social_Network_Ads.csv", sep=";")
    # normalize dataset
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    X = df[selected_input].to_numpy()
    y = df[selected_output[0]].to_numpy()

    # Split dataset
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(X, y, test_size=test_sz, random_state=random_st)

    print("Training data shape: ", X_train.shape, y_train.shape)
    print("validation data shape: ", X_test.shape, y_test.shape)
    print()

    # Train model
    logreg = LogisticRegression(random_state=random_st)
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

    result.append(train_result)
    result.append(test_result)

    data_shape.append(X_train.shape)
    data_shape.append(y_train.shape)

    data_shape.append(y_test.shape)
    data_shape.append(y_test.shape)

    return result, data_shape


df = pd.read_csv("data/Social_Network_Ads.csv", sep=";")
selected_input = ["Age", "EstimatedSalary"]
selected_output = ["Purchased"]
test_sz = 0.2
random_st = 19521338

result, data_shape = get_result_lg(
    df, selected_input, selected_output, test_sz, random_st
)

print("Training result:", result[0])
print("Test result:", result[1])
print()
print(f"Training shape: {str(data_shape[0])}, {str(data_shape[1])}")
print(f"Test shape: {str(data_shape[2])}, {str(data_shape[3])}")
