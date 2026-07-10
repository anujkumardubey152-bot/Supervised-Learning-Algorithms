# Logistic Regression and Decision Tree Classification

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

# Load dataset
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================
# Logistic Regression
# ============================

log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("========== Logistic Regression ==========")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall:", recall_score(y_test, log_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, log_pred))

# ============================
# Decision Tree
# ============================

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

tree_pred = tree_model.predict(X_test)

print("\n========== Decision Tree ==========")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print("Precision:", precision_score(y_test, tree_pred))
print("Recall:", recall_score(y_test, tree_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, tree_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, tree_pred))