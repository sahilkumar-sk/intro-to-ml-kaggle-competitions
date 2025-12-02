# Classification Kaggle Competition

This folder contains my work for a private **classification** competition completed as part of the *Introduction to Machine Learning* course at IBA.

---

## Overview  
The objective of the competition was to correctly classify rows into the target category.  
I developed full ML workflows involving:

- data preprocessing  
- feature engineering  
- model experimentation  
- hyperparameter tuning  
- PCA-based dimensionality reduction  
- evaluation using Accuracy, Precision, Recall, and F1-score  

---

## Models Experimented With

### **Decision Tree Classifier**  
- Default accuracy ≈ 70%  
- After tuning (`max_depth`, `min_samples_split`): **89%**

### **Naïve Bayes**  
- Initial ≈ 83%  
- Tried Laplace smoothing + PCA  
- Best: **84.4%**

### **K-Nearest Neighbors (KNN)**  
- Initial ≈ 54%  
- PCA(5) + k=11 improved accuracy to **60%**

### **Random Forest Classifier**  
- Base ≈ 85%  
- Tuning (`n_estimators`, `max_depth`) → **89%**

### **Extra Trees Classifier**  
- Base ≈ 83%  
- With `n_estimators=50`, `max_depth=10` → **88%**

### **Bagging Classifier**  
- Decision tree base ≈ 61%  
- Using RF / Extra Trees as base → **86.9%**

### **AdaBoost Classifier**  
- Strong performance with weak learners  
- Best accuracy: **94.9%** using:  
  - `n_estimators=300`  
  - `learning_rate=1`  
  - Base learner: Decision Tree (depth=1)

### **Voting Classifier**  
- Initial ≈ 72%  
- Best performance using **XGBoost + LightGBM + Random Forest**  
- PCA improved training speed

### **Stacking Classifier**  
- Initial ≈ 86.5%  
- Improved to **87.1%** using tuned base learners + GBoost meta learner

### **Gradient Boosting Classifier**  
- Initial ≈ 70.9%  
- With tuning (`n_estimators=100`, depth=5) → **90.9%**

### **XGBoost Classifier**  
- Initial ≈ 83.8%  
- Consistent improvement after tuning estimators + learning rate  
- **Final best-performing model** (highest competition score):

```python
n_estimators = 200
learning_rate = 0.1
max_depth = 5
min_child_weight = 1
gamma = 0
subsample = 0.8
colsample_bytree = 0.8
scale_pos_weight = 1
use_label_encoder = False
eval_metric = 'logloss'
