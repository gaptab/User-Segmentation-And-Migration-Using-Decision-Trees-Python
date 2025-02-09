import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Generate Dummy Data
np.random.seed(42)
num_samples = 5000

data = pd.DataFrame({
    'age': np.random.randint(20, 65, num_samples),
    'income': np.random.randint(25000, 150000, num_samples),
    'credit_score': np.random.randint(300, 850, num_samples),
    'loan_amount': np.random.randint(1000, 50000, num_samples),
    'loan_status': np.random.choice(['Paid', 'Defaulted', 'Ongoing'], num_samples),
    'active_loans': np.random.randint(0, 5, num_samples),
    'app_usage_frequency': np.random.randint(1, 30, num_samples),
    'engagement_score': np.random.randint(1, 100, num_samples),
    'migrated': np.random.choice([0, 1], num_samples, p=[0.98, 0.02])  # Target variable
})

# Step 2: Encode Categorical Feature ('loan_status')
data = pd.get_dummies(data, columns=['loan_status'], drop_first=True)

# Step 3: Define Features and Target Variable
X = data.drop(columns=['migrated'])  # Features
y = data['migrated']                 # Target

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Train Decision Tree Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)

# Step 7: Identify Valuable User Segments
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print("\nTop Factors Influencing Migration:\n", feature_importance)

# Step 8: Save Model & Data for Deployment
data.to_csv("user_segmentation_data.csv", index=False)
joblib.dump(model, "user_migration_model.pkl")

print("Dataset and Model Saved Successfully!")
