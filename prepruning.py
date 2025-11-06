import pandas as pd

df = pd.read_csv('heart.csv')

X = df.drop('target', axis=1)
Y = df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

dt_simple = DecisionTreeClassifier(random_state=42)
dt_simple.fit(X_train, y_train)
y_pred_simple = dt_simple.predict(X_test)
print(accuracy_score(y_test, y_pred_simple))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(
    dt_simple,
    feature_names=X.columns.tolist(),
    class_names=['Low Risk (0)', 'High Risk (1)'],
    filled=True
)
plt.plot()

MAX_DEPTH_VALUE = 3

dt_pruned = DecisionTreeClassifier(
    max_depth=MAX_DEPTH_VALUE,
    random_state=42
)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)
print(accuracy_score(y_test, y_pred_pruned))

plt.figure(figsize=(20, 10))
plot_tree(
    dt_pruned,
    feature_names=X.columns.tolist(),
    class_names=['Low Risk (0)', 'High Risk (1)'],
    filled=True
)
plt.plot()


