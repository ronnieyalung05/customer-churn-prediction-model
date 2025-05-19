# ALL NECESSARY IMPORTS ARE HERE
# DON'T FORGET TO LOAD THIS PART
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import matplotlib.ticker as mtick
warnings.filterwarnings('ignore')

# Loading in the dataset...
df = pd.read_csv("customer_churn.csv") # the data we'll be altering
DATA = df.copy() # un-altered data
# Show a few rows
DATA.head()


'''

PREPROCESS DATA

'''
# Check for missing values
print("\nMissing Values Before Preprocessing:")
print(df.isnull().sum())

# 1) Handle missing values in each column

# Customer_ID - Drop as it's not useful for prediction
df.drop('Customer_ID', axis=1, inplace=True)

# Gender - Convert to binary and handle missing values
# Create mapping dictionary
gender_map = {'M': 1, 'F': 0}
# Apply mapping to non-null values
df['Gender'] = df['Gender'].map(gender_map)
# Fill missing with mode (most common value)
gender_mode = df['Gender'].mode()[0]
df['Gender'] = df['Gender'].fillna(gender_mode)

# Age - Fill missing with median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# Signup_Date - Convert to datetime and handle missing values
df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], errors='coerce')
# Fill missing with median date
median_signup = df['Signup_Date'].dropna().median()
df['Signup_Date'] = df['Signup_Date'].fillna(median_signup)

# Last_Purchase - Convert to datetime and handle missing values
df['Last_Purchase'] = pd.to_datetime(df['Last_Purchase'], errors='coerce')
# Fill missing with median date
median_purchase = df['Last_Purchase'].dropna().median()
df['Last_Purchase'] = df['Last_Purchase'].fillna(median_purchase)

# Total_Spend - Fill missing with median
median_spend = df['Total_Spend'].median()
df['Total_Spend'] = df['Total_Spend'].fillna(median_spend)

# Payment_Method - Fill missing values before one-hot encoding
# Get most common payment method
most_common_payment = df['Payment_Method'].mode()[0]
df['Payment_Method'] = df['Payment_Method'].fillna(most_common_payment)

# Subscription_Type - Fill missing values before one-hot encoding
most_common_subscription = df['Subscription_Type'].mode()[0]
df['Subscription_Type'] = df['Subscription_Type'].fillna(most_common_subscription)

# Contract - Fill missing values before one-hot encoding
most_common_contract = df['Contract'].mode()[0]
df['Contract'] = df['Contract'].fillna(most_common_contract)

# Support_Calls - Fill missing with 0 (assuming no calls)
df['Support_Calls'] = df['Support_Calls'].fillna(0)

# Monthly_Visits - Fill missing with median
median_visits = df['Monthly_Visits'].median()
df['Monthly_Visits'] = df['Monthly_Visits'].fillna(median_visits)

# Churn - Our target variable
# Delete missing data; we don't want false positives or negatives
df.dropna(subset=['Churn'], inplace=True)

# 2) Feature engineering and encoding

# Create date-based features
current_date = df['Last_Purchase'].max()
df['Days_Since_Signup'] = (current_date - df['Signup_Date']).dt.days
df['Days_Since_Last_Purchase'] = (current_date - df['Last_Purchase']).dt.days

# One-hot encode Payment_Method
payment_dummies = pd.get_dummies(df['Payment_Method'], prefix='Payment')
df = pd.concat([df, payment_dummies], axis=1)
df.drop('Payment_Method', axis=1, inplace=True)

# One-hot encode Subscription_Type
subscription_dummies = pd.get_dummies(df['Subscription_Type'], prefix='Subscription')
df = pd.concat([df, subscription_dummies], axis=1)
df.drop('Subscription_Type', axis=1, inplace=True)

# One-hot encode Contract
contract_dummies = pd.get_dummies(df['Contract'], prefix='Contract')
df = pd.concat([df, contract_dummies], axis=1)
df.drop('Contract', axis=1, inplace=True)

# 3) Final verification and dataset review

# Check for any remaining missing values (should be none)
print("\nMissing Values After Preprocessing:")
print(df.isnull().sum())

# Display the first few rows of processed data
print("\nProcessed Data Preview:")
print(df.head())


'''

TRAIN MODEL

'''
# Step 1: Train/Test Split

# drop the datetime columns as they're not needed for modeling
# (we already created derived features from them)
df = df.drop(['Signup_Date', 'Last_Purchase'], axis=1)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.columns.tolist()}")
print(f"Churn distribution: {y.value_counts().to_dict()}")

# 1. Train/Test Split (80/20) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Scale numerical features
numeric_features = ['Age', 'Total_Spend', 'Support_Calls', 'Monthly_Visits', 
                   'Days_Since_Signup', 'Days_Since_Last_Purchase']
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# 2. Model Training

# a. Logistic Regression
print("\nTraining Logistic Regression Model...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
print("Logistic Regression model trained.")

# b. Decision Tree
print("\nTraining Decision Tree Model...")
# Basic model first
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Basic Decision Tree model trained.")

# Hyperparameter tuning for Decision Tree
print("\nPerforming hyperparameter tuning for Decision Tree...")
dt_param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt_grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=dt_param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

dt_grid_search.fit(X_train, y_train)
print(f"Best parameters for Decision Tree: {dt_grid_search.best_params_}")

# Get the best model
dt_best_model = dt_grid_search.best_estimator_

# c. Random Forest
print("\nTraining Random Forest Model...")
# Basic model first
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Basic Random Forest model trained.")

# Hyperparameter tuning for Random Forest
print("\nPerforming hyperparameter tuning for Random Forest...")
# Using RandomizedSearchCV for efficiency since the parameter space is larger
rf_param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'max_features': ['sqrt', 'log2', None]
}

rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_dist,
    n_iter=12,  # Number of parameter settings sampled
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

rf_random_search.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {rf_random_search.best_params_}")

# Get the best model
rf_best_model = rf_random_search.best_estimator_

print("\nAll models have been trained and tuned.")

'''

MODEL EVALUATION

'''

# Model evaluation function
def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and return metrics"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get the positive class label (may be '1' or 1 depending on data)
    # Find what classes are in the report
    classes = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
    pos_class = classes[-1]  # Usually the last class is positive
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve values
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    return {
        'name': model_name,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'report': report,
        'pos_class': pos_class,  # Store positive class label
        'confusion_matrix': cm,
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'pr': {'precision': precision, 'recall': recall}
    }

# 1. Evaluate all models
print("Evaluating model performance...\n")
models = {
    'Logistic Regression': lr_model,
    'Decision Tree (Tuned)': dt_best_model,
    'Random Forest (Tuned)': rf_best_model
}

model_results = {}
for name, model in models.items():
    model_results[name] = evaluate_model(model, X_test, y_test, name)
    pos_class = model_results[name]['pos_class']  # Get positive class label
    
    # Print key metrics
    print(f"Model: {name}")
    print(f"Accuracy: {model_results[name]['report']['accuracy']:.4f}")
    print(f"Precision (Churn={pos_class}): {model_results[name]['report'][pos_class]['precision']:.4f}")
    print(f"Recall (Churn={pos_class}): {model_results[name]['report'][pos_class]['recall']:.4f}")
    print(f"F1-Score (Churn={pos_class}): {model_results[name]['report'][pos_class]['f1-score']:.4f}")
    print(f"AUC-ROC: {model_results[name]['roc']['auc']:.4f}")
    print("-" * 50)

# 2. Create comparison table of model metrics
metrics_df = pd.DataFrame({
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': [],
    'AUC-ROC': []
})

for name, results in model_results.items():
    pos_class = results['pos_class']  # Get positive class label
    metrics_df = pd.concat([metrics_df, pd.DataFrame({
        'Model': [name],
        'Accuracy': [results['report']['accuracy']],
        'Precision': [results['report'][pos_class]['precision']],
        'Recall': [results['report'][pos_class]['recall']],
        'F1-Score': [results['report'][pos_class]['f1-score']],
        'AUC-ROC': [results['roc']['auc']]
    })], ignore_index=True)

print("\nModel Comparison Table:")
print(metrics_df.round(4))

# 3. Plot ROC curves
plt.figure(figsize=(10, 6))
for name, results in model_results.items():
    plt.plot(
        results['roc']['fpr'], 
        results['roc']['tpr'], 
        label=f"{name} (AUC = {results['roc']['auc']:.3f})"
    )

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Churn Prediction Models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 4. Plot confusion matrices
best_model_name = metrics_df.loc[metrics_df['F1-Score'].idxmax(), 'Model']
best_model = models[best_model_name]
best_results = model_results[best_model_name]

plt.figure(figsize=(10, 8))
cm = best_results['confusion_matrix']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

# Calculate percentages for annotations
total = cm.sum()
cm_percent = cm / total * 100

# Add percentage annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j + 0.5, i + 0.7, 
            f'({cm_percent[i, j]:.1f}%)', 
            ha='center', va='center', 
            color='black' if cm[i, j] < cm.max()/2 else 'white'
        )

category_names = ['Retained', 'Churned']
plt.xticks([0.5, 1.5], category_names)
plt.yticks([0.5, 1.5], category_names, rotation=0)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()

# 5. Feature Importance
def plot_feature_importance(model, model_name, feature_names):
    """Plot feature importance for a given model"""
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Create a dataframe for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'Feature Importance for {model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return importance_df

# Generate feature importance plots for all models
feature_importance_results = {}
for name, model in models.items():
    importance_df = plot_feature_importance(model, name, X.columns)
    if importance_df is not None:
        feature_importance_results[name] = importance_df
        
        print(f"\nTop 5 Features for {name}:")
        print(importance_df.head(5))

# 6. Churn Probability Distribution
plt.figure(figsize=(10, 6))
for name, results in model_results.items():
    sns.kdeplot(results['y_pred_proba'], label=name, fill=True, alpha=0.3)
    
plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Distribution of Churn Probabilities by Model')
plt.xlabel('Predicted Churn Probability')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()