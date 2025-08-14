import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

#loading the data set by using pandas & and then checking for the rows and col with grtting info of data types and bytes.

df = pd.read_csv("HeartDiseaseTrain-Test.csv")  
df.replace("?", np.nan, inplace=True) 
df.shape
df.columns.tolist()
df.info()
df.head()

#Converting the target variable to binary so that the col will be in int 0,1 form

df['target'] = df['target'].astype(int)
df['target'] = (df['target'] > 0).astype(int)

#Checking and dropping missing values so that the data will be good for molde

df.isna().sum()
if df.isna().sum().sum() > 0 and df.isna().sum().sum() < 10:
    df = df.dropna().reset_index(drop=True)
    print("dropped rows with missing values. New shape:", df.shape)

#Splitting features (X) and target (y) to divid the data for model
    
target_col = 'target'
X = df.drop(columns=[target_col])
y = df[target_col]

#Detecting numeric and categorical columns dynamically

possible_num_cols = ['age','trestbps','chol','thalach','oldpeak']
num_cols = [col for col in possible_num_cols if col in X.columns]
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

#Building preprocessing pipelines:preparing data for the model

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

#Combining pipelines with ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

#Splitting data into train/test

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20, stratify=y)

#Logistic Regression model pipeline

log_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=2000))
])
log_pipe.fit(X_train, y_train)

#Predictions and probabilities

y_pred = log_pipe.predict(X_test)
y_proba = log_pipe.predict_proba(X_test)[:, 1]  

print("\n=== Logistic Regression Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test,y_proba):.3f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

ohe = log_pipe.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
ohe_cols = ohe.get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, ohe_cols])

coeffs = log_pipe.named_steps['clf'].coef_[0]
feat_imp = pd.Series(coeffs, index=feature_names).sort_values(key=abs, ascending=False)
print("\nTop features (by absolute logistic coefficient):")
print(feat_imp.head(8))

#Decision Tree model & visualization

tree_pipe = Pipeline([
    ('pre', preprocessor), 
    ('clf', DecisionTreeClassifier(max_depth=4, random_state=42))
])
tree_pipe.fit(X_train, y_train)

y_pred_tree = tree_pipe.predict(X_test)
y_proba_tree = tree_pipe.predict_proba(X_test)[:,1]

print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_tree))

tree_importances = tree_pipe.named_steps['clf'].feature_importances_
imp_series = pd.Series(tree_importances, index=feature_names).sort_values(ascending=False)
print("\nDecision tree feature importances:")
print(imp_series.head(8))

plt.figure(figsize=(12,8))
plot_tree(tree_pipe.named_steps['clf'], feature_names=feature_names, class_names=['No','Yes'], filled=True, rounded=True, fontsize=8)
plt.show()
