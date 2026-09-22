import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 1. Load dataset
df = pd.read_csv("creditcard.csv")

# 2. Basic information
print(df.head())
print(df.shape)
print(df.isnull().sum())

# 3. Fraud/Genuine count
print("\nClass Distribution:")
print(df["Class"].value_counts())

# 4. Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# 8. Train model
model.fit(X_train_scaled, y_train)

# 9. Prediction
y_pred = model.predict(X_test_scaled)

# 10. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance")
print("----------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# 11. Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 13. New transaction
new_transaction = X.iloc[[0]]

new_transaction_scaled = scaler.transform(
    new_transaction
)

prediction = model.predict(
    new_transaction_scaled
)

if prediction[0] == 1:
    print("\n⚠️ Fraud Transaction")
else:
    print("\n✅ Genuine Transaction")
