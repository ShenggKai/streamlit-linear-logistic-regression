import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import metrics


def get_result_lg(df, selected_input, selected_output, test_sz, random_st):
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

    # Train model
    logreg = LogisticRegression(random_state=random_st)
    logreg.fit(X_train, y_train)

    train_result = round(
        f1_score(y_train, logreg.predict(X_train), average="weighted"), 5
    )

    test_result = round(
        f1_score(y_test, logreg.predict(X_test), average="weighted"), 5
    )

    # Create confusion matrix
    y_pred = logreg.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g")
    plt.tight_layout()
    plt.title("Confusion matrix", fontsize=16)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    # get F1-score values
    result = [train_result, test_result]
    # get data shape
    data_shape = [X_train.shape, y_train.shape, X_test.shape, y_test.shape]

    return result, data_shape, fig
