AI/ML Internship Task For DevelopersHub Corporation

This repository contains multiple tasks focusing on data analysis, visualization, predictive modeling, pipelines, multimodal ML, and LLM-based classification. Each task involves a specific dataset, objectives, applied models, and key findings.

Task 1: Exploring and Visualising a Simple Dataset
Objective

Understand how to load, inspect, and visualise a dataset to explore trends, distributions, and relationships between features.

Dataset

Name: Iris Dataset

Format: CSV (can also be loaded via seaborn.load_dataset("iris"))

Models Applied

No predictive model applied — this task focuses purely on Exploratory Data Analysis (EDA).

Key Results and Findings

Loaded and inspected the dataset using Pandas (.shape, .head(), .info(), .describe()).

Generated scatter plots to visualise feature relationships (e.g., petal length vs petal width).

Used histograms to understand value distributions.

Applied box plots to detect potential outliers.

Observed that species have distinct patterns in petal/sepal measurements.

Task 2: Heart Disease Prediction
Objective

Build and evaluate a machine learning model to predict whether a patient is at risk of heart disease based on their health metrics.

Dataset

Name: Heart Disease UCI Dataset

Source: Kaggle

Description: Contains various medical features such as age, cholesterol, resting blood pressure, and more.

Models Applied

Logistic Regression

Decision Tree Classifier

Key Results and Findings

Cleaned the dataset by handling missing values and performing basic preprocessing.

Conducted EDA to identify trends (e.g., higher cholesterol often correlates with increased heart disease risk).

Evaluated models using accuracy, ROC curve, and confusion matrix:

Best Accuracy: ~85% (Logistic Regression)

ROC-AUC Score: ~0.89

Identified key features influencing predictions:

Chest pain type (cp)

Maximum heart rate achieved (thalach)

ST depression induced by exercise (oldpeak)

Logistic Regression provided better generalization compared to the Decision Tree.

Task 3: House Price Prediction
Objective

Predict house prices using property features such as size, bedrooms, and location.

Dataset

Name: House Price Prediction Dataset

Source: Kaggle

Description: Contains property details like square footage, number of bedrooms, and city.

Models Applied

Linear Regression

Gradient Boosting Regressor

Key Results and Findings

Preprocessed features including square footage, number of bedrooms, and location (encoded categorical variables).

Visualized predicted prices versus actual prices for better interpretation.

Evaluated models using:

Mean Absolute Error (MAE): ~25,000

Root Mean Squared Error (RMSE): ~35,000

Gradient Boosting outperformed Linear Regression, capturing non-linear relationships better.

Task 4: End-to-End ML Pipeline with Scikit-learn Pipeline API
Objective

Build a reusable and production-ready machine learning pipeline for predicting customer churn.

Dataset

Name: Telco Churn Dataset

Instructions

Implement data preprocessing steps (e.g., scaling, encoding) using Pipeline.

Train models like Logistic Regression and Random Forest.

Use GridSearchCV for hyperparameter tuning.

Export the complete pipeline using joblib.

Skills Gained

ML pipeline construction

Hyperparameter tuning with GridSearch

Model export and reusability

Production-readiness practices

Task 5: Multimodal ML – Housing Price Prediction Using Images + Tabular Data
Objective

Predict housing prices using both structured data and house images.

Dataset

Name: Housing Sales Dataset + Custom Image Dataset (your own or any public source)

Instructions

Use CNNs to extract features from images.

Combine extracted image features with tabular data.

Train a model using both modalities.

Evaluate performance using MAE and RMSE.

Skills Gained

Multimodal machine learning

Convolutional Neural Networks (CNNs)

Feature fusion (image + tabular)

Regression modeling and evaluation

Task 6: Auto Tagging Support Tickets Using LLM
Objective

Automatically tag support tickets into categories using a large language model (LLM).

Dataset

Name: Free-text Support Ticket Dataset

Instructions

Use prompt engineering or fine-tuning with an LLM.

Compare zero-shot vs fine-tuned performance.

Apply few-shot learning techniques to improve accuracy.

Output top 3 most probable tags per ticket.

Skills Gained

Prompt engineering

LLM-based text classification

Zero-shot and few-shot learning

Multi-class prediction and ranking
