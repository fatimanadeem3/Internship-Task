AI/ML Internship Tasks – DevelopersHub Corporation

This repository contains six tasks focusing on data analysis, visualization, predictive modeling, pipelines, multimodal learning, and LLM-based classification.

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

Description: Contains medical features such as age, cholesterol, resting blood pressure, chest pain type, maximum heart rate, etc.

Models Applied

Logistic Regression

Decision Tree Classifier

Key Results and Findings

Cleaned the dataset and handled missing values.

Conducted EDA to identify patterns (e.g., high cholesterol linked to higher heart disease risk).

Evaluated models using:

Best Accuracy: ~85% (Logistic Regression)

ROC-AUC Score: ~0.89

Key predictive features included:

Chest pain type (cp)

Maximum heart rate (thalach)

ST depression induced by exercise (oldpeak)

Logistic Regression generalized better than the Decision Tree.

Task 3: House Price Prediction
Objective

Predict house prices using property features such as size, bedrooms, and location.

Dataset

Name: House Price Prediction Dataset

Source: Kaggle

Description: Contains property details like square footage, number of bedrooms, location, and price.

Models Applied

Linear Regression

Gradient Boosting Regressor

Key Results and Findings

Preprocessed features including square footage, bedrooms, and encoded location.

Visualized predicted vs actual prices.

Evaluation metrics:

Mean Absolute Error (MAE): ~25,000

Root Mean Squared Error (RMSE): ~35,000

Gradient Boosting outperformed Linear Regression by capturing non-linear patterns.

Task 4: End-to-End ML Pipeline with Scikit-learn Pipeline API
Objective

Build a reusable and production-ready machine learning pipeline for predicting customer churn.

Dataset

Name: Telco Churn Dataset

Models Applied

Logistic Regression

Random Forest

Key Results and Findings

Implemented scaling & encoding with the Pipeline API.

Used GridSearchCV for hyperparameter tuning.

Exported the pipeline with joblib for deployment.

Achieved high accuracy while ensuring production readiness.

Task 5: Multimodal ML – Housing Price Prediction Using Images + Tabular Data
Objective

Predict housing prices using both structured data and house images.

Dataset

Name: Housing Sales Dataset + Custom Image Dataset (public/self-collected).

Models Applied

Convolutional Neural Networks (CNNs) for image features

Fusion model (image + tabular features combined)

Key Results and Findings

Extracted image features with CNNs.

Combined them with tabular features for training.

Outperformed tabular-only models with better accuracy.

Evaluation metrics:

Mean Absolute Error (MAE): ~22,000

Root Mean Squared Error (RMSE): ~30,000

Task 6: Auto Tagging Support Tickets Using LLM
Objective

Automatically classify support tickets into categories using a Large Language Model (LLM).

Dataset

Name: Free-text Support Ticket Dataset

Models Applied

Pre-trained LLM with prompt engineering

Zero-shot and Few-shot learning

Key Results and Findings

Compared zero-shot vs fine-tuned performance.

Used few-shot examples to improve classification accuracy.

Generated top 3 probable tags per ticket.

Achieved strong results for real-world multi-class support ticket classification.

Skills Gained Across Tasks

Data Analysis & EDA → Pandas, Matplotlib, Seaborn

Classification Models → Logistic Regression, Decision Tree

Regression Models → Linear Regression, Gradient Boosting

Pipelines → Scikit-learn Pipeline API, GridSearchCV, joblib

Deep Learning → CNNs for images, multimodal fusion

LLMs → Prompt engineering, zero-shot & few-shot classification

Evaluation Metrics → Accuracy, ROC curve, Confusion Matrix, MAE, RMSE
