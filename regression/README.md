# Kaggle Regression Competition 

This folder contains my work for a private regression competition completed as part of the **Introduction to Machine Learning** course at IBA.

## 📌 Overview
The goal of the competition was to predict a continuous target variable from a tabular dataset.  
I built full ML workflows including:
- data preprocessing  
- feature engineering  
- model experimentation  
- hyperparameter tuning  
- evaluation using RMSE, MAE, and R²  

## 🧪 Models Experimented With
I tested and compared multiple algorithms:

- **Linear Regression / Ridge Regression**
- **KNN Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **AdaBoost Regressor**
- **Gradient Boosting Regressor**
- **XGBoost Regressor**
- (Attempted but not used due to constraints: Stacking, Neural Networks, Forward/Backward selection)

I also applied **PCA** for dimensionality reduction and performance improvement.

## 🏆 Best Model
The best performance came from **XGBoost Regressor** with the following tuned hyperparameters:

```python
n_estimators = 1200
learning_rate = 0.003
max_depth = 13
subsample = 0.85
colsample_bytree = 0.9
