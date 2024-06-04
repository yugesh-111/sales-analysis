import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Load your dataset
df = pd.read_csv('Dummy Data HSS.csv')

df.isna().any(axis=1).sum()
df = df.dropna(axis=0)

# Discretize the sales data
# Example: Create categories based on quantiles
df['SalesCategory'] = pd.qcut(df['Sales'], q=3, labels=['Low', 'Medium', 'High'])

# Feature and target selection
X = df[['TV', 'Radio']]
y = df['SalesCategory']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model
log_regressor = LogisticRegression(max_iter=1000)

# Train the model
log_regressor.fit(X_train, y_train)

# Make predictions
y_pred_log = log_regressor.predict(X_test)

# Evaluate the model
accuracy_log = accuracy_score(y_test, y_pred_log)
precision_log = precision_score(y_test, y_pred_log, average='weighted')
recall_log = recall_score(y_test, y_pred_log, average='weighted')
f1_log = f1_score(y_test, y_pred_log, average='weighted')
log_loss_log = log_loss(y_test, log_regressor.predict_proba(X_test))
# Initialize the model
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
log_loss_dt = log_loss(y_test, dt_classifier.predict_proba(X_test))


# Initialize the model
rf_classifier = RandomForestClassifier(random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
log_loss_rf = log_loss(y_test, rf_classifier.predict_proba(X_test))


# Print results in a tabular format
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_log, accuracy_dt, accuracy_rf],
    'Precision': [precision_log, precision_dt, precision_rf],
    'Recall': [recall_log, recall_dt, recall_rf],
    'F1 Score': [f1_log, f1_dt, f1_rf],
    'Log Loss': [log_loss_log, log_loss_dt, log_loss_rf]
})

print(results)
