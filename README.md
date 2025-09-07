# AI/ML Internship Tasks For DevelopersHub Corporation

This repository contains multiple tasks focusing on data analysis, visualization, predictive modeling, and advanced machine learning approaches. Each task involves a specific dataset, objectives, applied models, and key findings.

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

## **Task 4: Auto Tagging Support Tickets Using LLM**

### **Objective**
Automatically tag support tickets into categories using a large language model (LLM).

### **Dataset**
- **Name:** Free-text Support Ticket Dataset
- **Format:** Plain text / CSV
- **Source:** https://www.kaggle.com/akashbommidi

### **Models Applied**
- Large Language Model (LLM) – Zero-shot, Few-shot, Fine-tuned approaches

### **Key Results and Findings**
- Used **prompt engineering** for zero-shot classification.
- Applied **few-shot learning** to improve accuracy.
- Compared zero-shot vs fine-tuned performance.
- Produced **top 3 most probable tags** for each ticket.

---

## **Task 5: End-to-End ML Pipeline with Scikit-learn Pipeline API**

### **Objective**
Build a reusable and production-ready machine learning pipeline for predicting customer churn.

### **Dataset**
- **Name:** Telco Churn Dataset
- **Source:** Public telecom dataset
- **Description:** Contains customer demographics, account details, and churn labels.

### **Models Applied**
- Logistic Regression
- Random Forest

### **Key Results and Findings**
- Implemented preprocessing (scaling, encoding) inside a **Pipeline**.
- Applied **GridSearchCV** for hyperparameter tuning.
- Exported the final pipeline using **joblib** for production use.
- Random Forest outperformed Logistic Regression on test data.

---

## **Task 6: Multimodal ML – Housing Price Prediction Using Images + Tabular Data**

### **Objective**
Predict housing prices using both structured tabular data and house images.

### **Dataset**
- **Name:** Housing Sales Dataset + Custom Image Dataset
- **Source:** [Public dataset or custom-collected images](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
- **Description:** Includes property details (square footage, bedrooms, location) and images of houses.

### **Models Applied**
- Convolutional Neural Networks (CNNs) for image feature extraction
- Regression models combining image + tabular features

### **Key Results and Findings**
- Extracted image embeddings using CNN.
- Combined image features with tabular property data.
- Trained regression models on the fused features.
- Evaluated performance using:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
- Multimodal fusion significantly outperformed single-modality models.

---

## **Skills Gained Across Tasks**
- Data loading, inspection, and cleaning using Pandas
- Exploratory Data Analysis (EDA) using Matplotlib & Seaborn
- Binary classification using Logistic Regression & Decision Tree
- Regression modeling using Linear Regression, Gradient Boosting, and Random Forest
- LLM-based text classification (Zero-shot, Few-shot, Fine-tuning)
- Multimodal learning: combining CNN image features with tabular data
- Model evaluation using accuracy, ROC curve, confusion matrix, MAE, and RMSE
- Production-ready ML pipelines with **Scikit-learn Pipeline API** and **joblib**
