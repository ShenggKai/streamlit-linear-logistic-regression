import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_result_ln(df, selected_input, selected_output, test_sz, random_st):
    # normalize dataset
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    X = df[selected_input]
    y = df[selected_output[0]].values

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=random_st
    )

    # Training model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)

    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    mae_train = mean_absolute_error(y_true=y_train, y_pred=y_pred_train)
    mse_train = mean_squared_error(y_true=y_train, y_pred=y_pred_train)
    rmse_train = mean_squared_error(
        y_true=y_train, y_pred=y_pred_train, squared=False
    )

    mae_test = mean_absolute_error(y_true=y_test, y_pred=y_pred_test)
    mse_test = mean_squared_error(y_true=y_test, y_pred=y_pred_test)
    rmse_test = mean_squared_error(
        y_true=y_test, y_pred=y_pred_test, squared=False
    )

    result_train = [train_score, mae_train, mse_train, rmse_train]
    result_test = [test_score, mae_test, mse_test, rmse_test]
    data_shape = [X_train.shape, y_train.shape, X_test.shape, y_test.shape]

    return result_train, result_test, data_shape
