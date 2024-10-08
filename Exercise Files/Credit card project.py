import pandas as pd
file_path = r"C:\Users\emmanuel.elias\OneDrive - FE fundinfo\Desktop\Credit_Card_Fraud_Detection.xlsx"
df = pd.read_excel(r"C:\Users\emmanuel.elias\OneDrive - FE fundinfo\Desktop\Credit_Card_Fraud_Detection.xlsx")
df.head()

df.isnull().sum()

df.describe()

df['class'].value_counts(normalize=True)

from imblearn.over_sampling import SMOTE

X = df.drop(columns=['class'])
y = df['class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

pd.Series(y_resampled).value_counts(normalize=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred_rf = model_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

roc_auc_rf = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score (Random Forest): {roc_auc_rf}")

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt 

fpr, tpr, _ = roc_curve(y_test, model_rf.predict_proba(X_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc_rf:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(model_rf, param_grid, cv=3, scoring='roc_auc')
grid_search_rf.fit(X_train, y_train)

grid_search_rf.best_params_

