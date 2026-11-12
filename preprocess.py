import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv('data/results.csv')
drivers = pd.read_csv('data/drivers.csv')
races = pd.read_csv('data/races.csv')
constructors = pd.read_csv('data/constructors.csv')

df = results.merge(races[['raceId','year','circuitId']], on='raceId', how='left')
df = df.merge(drivers[['driverId','nationality']], on='driverId', how='left')
df = df.merge(constructors[['constructorId','name']], on='constructorId', how='left')

df['position'] = pd.to_numeric(df['position'], errors='coerce')
df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['laps'] = pd.to_numeric(df['laps'], errors='coerce')
df['fastestLap'] = pd.to_numeric(df['fastestLap'], errors='coerce')
df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
df['fastestLapSpeed'] = pd.to_numeric(df['fastestLapSpeed'], errors='coerce')

df['statusId'] = pd.to_numeric(df['statusId'], errors='coerce')
df['top3'] = df['position'].apply(lambda x: 1 if x <= 3 else 0)

features = ['grid', 'laps', 'fastestLap', 'rank', 'fastestLapSpeed', 'year', 'statusId']
target = 'top3'

df_clean = df[features + [target]].dropna()

le_nat = LabelEncoder()
df_clean['nationality_encoded'] = LabelEncoder().fit_transform(
    df.loc[df_clean.index, 'nationality'].fillna('Unknown')
)

le_con = LabelEncoder()
df_clean['constructor_encoded'] = LabelEncoder().fit_transform(
    df.loc[df_clean.index, 'name'].fillna('Unknown')
)

features_final = features + ['nationality_encoded', 'constructor_encoded']
X = df_clean[features_final]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

importances = pd.Series(rf.feature_importances_, index=features_final).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index)
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("Plot saved: feature_importance.png")
