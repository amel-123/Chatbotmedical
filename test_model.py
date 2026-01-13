import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Charger les fichiers
training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

cols = training.columns[:-1]
x_train = training[cols]
y_train = preprocessing.LabelEncoder().fit_transform(training['prognosis'])
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(training['prognosis'])

x_test = testing[cols]
y_test = le.transform(testing['prognosis'])

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# Tester le modèle
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on Testing.csv: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
