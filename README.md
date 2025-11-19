# 🩺 Diabetes Prediction using Machine Learning
This project builds and compares multiple machine-learning models to predict diabetes using the **Pima Indians Diabetes Dataset**.
It includes data preprocessing, visualization, model training, evaluation, and accuracy comparison.

## 📂 Dataset
**Pima Indians Diabetes Dataset**
- Rows: 768
- Features: 8 clinical measurements
- Target: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

## 🧠 Models Used & Explanation
### Logistic Regression
Baseline linear classifier.

### K-Nearest Neighbors (KNN)
Distance-based non-parametric model.

### Support Vector Classifier (SVC)
Maximizes the decision boundary margin.

### Random Forest
Ensemble of decision trees.

### XGBoost
Gradient boosting model — best performer.

## 📊 Results Summary
| Model | Best Accuracy |
|-------|--------------|
| Logistic Regression | ~78% |
| KNN | ~75% |
| SVC | ~80% |
| Random Forest | ~86% |
| **XGBoost** | **~88% (Best Performer)** |

## 📉 Visualizations Included
- Correlation heatmap  
- Pairplot  
- Accuracy comparison bar graph  
- Confusion matrices  
- ROC curves  
- Feature importance plots  

## 🛠️ Tech Stack
Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn, Google Colab

## ▶️ How to Run
Install dependencies:
```
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

## 🧪 Workflow
1. Load dataset  
2. Preprocess  
3. Visualize  
4. Train models  
5. Tune hyperparameters  
6. Compare results  

## 🏆 Conclusion
XGBoost achieved the highest accuracy (~88%).

