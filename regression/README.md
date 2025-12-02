# Regression Kaggle Competition 

This folder contains my work for a private **regression** competition completed as part of the *Introduction to Machine Learning* course at IBA.

---

## Overview  
The goal of the competition was to predict a continuous target variable using a tabular dataset.  
I developed complete ML pipelines involving:

- data preprocessing  
- feature engineering  
- model experimentation  
- hyperparameter tuning  
- PCA-based dimensionality reduction  
- evaluation using RMSE, MAE, and R²  

---

## Models Experimented With

### **Linear Regression & Ridge Regression**  
- Baseline performance models  
- Ridge (α = 1.5) performed better than plain linear regression

### **K-Nearest Neighbors (KNN) Regression**  
- Initial performance was weak  
- PCA (60 components) + k=15 produced best KNN performance

### **Decision Tree Regressor**  
- Base model was unstable  
- Best performance around `max_depth=6`

### **Random Forest Regressor**  
- Strong baseline  
- Tuned parameters improved performance significantly:  
  - `n_estimators=100`  
  - `max_depth=10`  
  - `min_samples_split=20`  
  - `min_samples_leaf=5`

### **AdaBoost Regressor**  
- Used decision tree as weak learner  
- Best combination:  
  - `n_estimators=100`  
  - `learning_rate=0.3`  
  - Base tree depth = 6

### **Gradient Boosting Regressor**  
- Improved over AdaBoost  
- Best at:  
  - `n_estimators=200`  
  - `learning_rate=0.05`  
  - `max_depth=4`

### **XGBoost Regressor**  
- Consistently delivered the strongest results  
- Required tuning due to dataset complexity  
- Best model performance achieved with:

```python
n_estimators = 1200
learning_rate = 0.003
max_depth = 13
subsample = 0.85
colsample_bytree = 0.9
