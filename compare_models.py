import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import time

results = pd.read_csv('data/results.csv')
drivers = pd.read_csv('data/drivers.csv')
races = pd.read_csv('data/races.csv')
constructors = pd.read_csv('data/constructors.csv')

df = results.merge(races[['raceId','year','circuitId']], on='raceId', how='left')
df = df.merge(drivers[['driverId','nationality']], on='driverId', how='left')
df = df.merge(constructors[['constructorId','name']], on='constructorId', how='left')

for col in ['position','grid','points','laps','fastestLap','rank','fastestLapSpeed','statusId']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['top3'] = df['position'].apply(lambda x: 1 if x <= 3 else 0)

features = ['grid','laps','fastestLap','rank','fastestLapSpeed','year','statusId']
target = 'top3'

df_clean = df[features + [target]].dropna()
df_clean['nationality_encoded'] = LabelEncoder().fit_transform(df.loc[df_clean.index, 'nationality'].fillna('Unknown'))
df_clean['constructor_encoded'] = LabelEncoder().fit_transform(df.loc[df_clean.index, 'name'].fillna('Unknown'))

features_final = features + ['nationality_encoded', 'constructor_encoded']
X = df_clean[features_final]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
}

results_dict = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results_dict[name] = {'accuracy': acc, 'time': train_time}
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    print(classification_report(y_test, y_pred))

print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)
for name, res in sorted(results_dict.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:20s} | Accuracy: {res['accuracy']:.4f} | Time: {res['time']:.2f}s")
