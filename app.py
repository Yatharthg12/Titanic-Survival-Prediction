import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# loading data
data = pd.read_csv("train.csv")
print("Data loaded:", data.shape)
print(data.head(3))

# filling missing values manually (a bit safer this way)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# not useful cols
data = data.drop(['Cabin', 'Name', 'Ticket'], axis=1)

# encoding categorical stuff
enc = LabelEncoder()
data['Sex'] = enc.fit_transform(data['Sex'])
data['Embarked'] = enc.fit_transform(data['Embarked'])

# separating features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# splitting data
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {Xtrain.shape}, Test size: {Xtest.shape}")

# scaling
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# training three models
models = {
    'LogReg': LogisticRegression(max_iter=500),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

print("\nTraining models...\n")
for name, model in models.items():
    model.fit(Xtrain_scaled, ytrain)
    preds = model.predict(Xtest_scaled)
    acc = accuracy_score(ytest, preds)
    cm = confusion_matrix(ytest, preds)
    results[name] = {'acc': acc, 'cm': cm}
    print(f"{name} -> accuracy: {acc:.4f}")

# accuracy table
acc_df = pd.DataFrame([(n, round(v['acc'], 4)) for n, v in results.items()],
                      columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)

print("\nAccuracy Comparison:")
print(acc_df.to_string(index=False))

# showing all confusion matrices together
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, metrics) in zip(axes, results.items()):
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# feature importance (just for RandomForest)
rf = models['RandomForest']
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(7, 4))
sns.barplot(x=imp.values, y=imp.index, palette='mako')
plt.title('Feature Importance (RandomForest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# print top performing model
best_model = acc_df.iloc[0]
print(f"\nBest model overall: {best_model['Model']} ({best_model['Accuracy']:.4f})")
