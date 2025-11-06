import pandas as pd

df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(report_lr)

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train_scaled, y_train)
y_pred_rf = rf_clf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(report_rf)

new_patient_data = pd.DataFrame([
    {'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
     'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3,
     'slope': 0, 'ca': 0, 'thal': 1}
    ])

new_patient_scaled = scaler.transform(new_patient_data)

prediction = rf_clf.predict(new_patient_scaled)[0]
risk_status = "High Risk (Class 1)" if prediction == 1 else "Low Risk (Class 0)"

print(f"Predicted Risk Status: {risk_status}")
