import pandas as pd

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis=1)
Y = df['Outcome']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled)
X.head(4)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

clf = BaggingClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print('BAGGING RESULT =')
print('Accuracy : ', accuracy, '\nPrecision : ', precision, '\nRecall : ', recall, '\nF1-Score : ', f1)

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier().fit(x_train, y_train)
y_pred = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print('ADABOOST RESULT =')
print('Accuracy : ', accuracy, '\nPrecision : ', precision, '\nRecall : ', recall, '\nF1-Score : ', f1)

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

estimators = [
    ('svm', SVC(probability=True)),
    ('bayes', GaussianNB()),
    ('knn', KNeighborsClassifier()),
    ('cart', DecisionTreeClassifier())
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5
).fit(x_train, y_train)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print('STACKING RESULT =')
print('Accuracy : ', accuracy, '\nPrecision : ', precision, '\nRecall : ', recall, '\nF1-Score : ', f1)

