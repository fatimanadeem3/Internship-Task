AI/ML Internship Task For DevelopersHub Corporation

This repository contains three separate tasks focusing on data analysis, visualization, and predictive modeling. Each task involves a specific dataset, objectives, applied models, and key findings.

---

## **Task 1: Exploring and Visualising a Simple Dataset**

### **Objective**
Understand how to load, inspect, and visualise a dataset to explore trends, distributions, and relationships between features.

### **Dataset**
- **Name:** Iris Dataset  
- **Format:** CSV (can also be loaded via `seaborn.load_dataset("iris")`)

### **Models Applied**
- No predictive model applied — this task focuses purely on **Exploratory Data Analysis (EDA)**.

### **Key Results and Findings**
- Loaded and inspected the dataset using Pandas (`.shape`, `.head()`, `.info()`, `.describe()`).
- Generated scatter plots to visualise feature relationships (e.g., petal length vs petal width).
- Used histograms to understand value distributions.
- Applied box plots to detect potential outliers.
- Observed that species have distinct patterns in petal/sepal measurements.

---

## **Task 2: Heart Disease Prediction**

### **Objective**
Build and evaluate a machine learning model to predict whether a patient is at risk of heart disease based on their health metrics.

### **Dataset**
- **Name:** Heart Disease UCI Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/ketangangal/heart-disease-dataset-uci)  
- **Description:** Contains various medical features such as age, cholesterol, resting blood pressure, and more.

### **Models Applied**
- Logistic Regression
- Decision Tree Classifier

### **Key Results and Findings**
- Cleaned the dataset by handling missing values and performing basic preprocessing.
- Conducted EDA to identify trends (e.g., higher cholesterol often correlates with increased heart disease risk).
- Evaluated models using accuracy, ROC curve, and confusion matrix:
  - **Best Accuracy:** ~85% (Logistic Regression)
  - **ROC-AUC Score:** ~0.89
- Identified key features influencing predictions:
  - Chest pain type (cp)
  - Maximum heart rate achieved (thalach)
  - ST depression induced by exercise (oldpeak)
- Logistic Regression provided better generalization compared to the Decision Tree.

---

## **Task 3: House Price Prediction**

### **Objective**
Predict house prices using property features such as size, bedrooms, and location.

### **Dataset**
- **Name:** House Price Prediction Dataset  
- **Source:** [Kaggle](https://www.kaggle.com/datasets/shree1992/housedata)  
- **Description:** Contains property details like square footage, number of bedrooms, and city.

### **Models Applied**
- Linear Regression
- Gradient Boosting Regressor

### **Key Results and Findings**
- Preprocessed features including square footage, number of bedrooms, and location (encoded categorical variables).
- Visualized predicted prices versus actual prices for better interpretation.
- Evaluated models using:
  - **Mean Absolute Error (MAE):** ~25,000
  - **Root Mean Squared Error (RMSE):** ~35,000
- Gradient Boosting outperformed Linear Regression, capturing non-linear relationships better.

---

## **Skills Gained Across Tasks**
- Data loading, inspection, and cleaning using Pandas
- Exploratory Data Analysis (EDA) using Matplotlib & Seaborn
- Binary classification using Logistic Regression & Decision Tree
- Regression modeling using Linear Regression & Gradient Boosting
- Model evaluation using accuracy, ROC curve, confusion matrix, MAE, and RMSE
- Feature importance analysis
