Source for data: https://www.kaggle.com/datasets/yasserh/titanic-dataset

Procedure:
1. Data Cleaning:
  (i)Filled missing values in Age and Embarked
  (ii)Dropped irrelevant or high-missing columns (Cabin, Ticket, Name)

2. Encoding & Scaling: 
  (i)Converted categorical columns (Sex, Embarked) into numerical form
  (ii)Standardized numeric features for model stability

3. Model Training: Three machine learning models were trained - RandomForest, LogisticRegression and DecisionTree

4. Display Results:
   (i)Compare Model Accuracies
   (ii)Plotted confusion matrices for all models
   (iii)Displayed feature importance for Random Forest

Results: (Accuracies)
RandomForest - 0.83
LogisticRegression - 0.81
DecisionTree - 0.76

Best Performing Model - RandomForest (0.83 accuracy)

Required Libraries for Installation: pandas numpy scikit-learn matplotlib seaborn

Author:
Yatharth Garg
