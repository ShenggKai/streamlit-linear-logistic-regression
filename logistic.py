# import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# from sklearn import metrics
from sklearn.metrics import f1_score

# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

df = pd.read_csv("data/Social_Network_Ads.csv", sep=";")
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

"""## Dataset đã chuẩn hóa"""
x = df.iloc[:, :2].to_numpy()
y = df["Purchased"].to_numpy()

(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(x, y, test_size=0.2, random_state=19521338)

print()
print("Training data shape: ", X_train.shape, y_train.shape)
print()
print("Test data shape: ", X_test.shape, y_test.shape)

print()
print("Training...")
print()
logreg = LogisticRegression(random_state=19521338)
logreg.fit(X_train, y_train)
print("Done")

print()
print("Training set")
print(
    "- F1 score = %.5f"
    % f1_score(
        y_train,
        logreg.predict(X_train),
        average="weighted",
    )
)

print()
print("Test set")
print(
    "- F1 score = %.5f"
    % f1_score(
        y_test,
        logreg.predict(X_test),
        average="weighted",
    )
)

# """## Dataset đã chuẩn hóa"""

# y_pred = logreg.predict(X_test)
# cnf_matrix = metrics.confusion_matrix(
#     y_test, y_pred
# )
# cnf_matrix

# class_names = [0, 1]  # name  of classes
# fig, ax = plt.subplots()
# tick_marks = np.arange(len(class_names))
# plt.xticks(tick_marks, class_names)
# plt.yticks(tick_marks, class_names)
# # create heatmap
# sns.heatmap(
#     pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt="g"
# )
# plt.tight_layout()
# plt.title("Confusion matrix", fontsize=16)
# plt.ylabel("Actual label")
# plt.xlabel("Predicted label")
