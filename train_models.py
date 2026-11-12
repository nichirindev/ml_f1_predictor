"""
Train RF, Gradient Boosting, XGBoost on F1 data (predict podium top-3).
Saves models + encoders to models/ for later inference.
Usage: python train_models.py
"""
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES_NUM = ['grid', 'laps', 'fastestLap', 'rank', 'fastestLapSpeed', 'year', 'statusId']
TARGET = 'top3'


def load_data():
    results = pd.read_csv(DATA_DIR / 'results.csv')
    drivers = pd.read_csv(DATA_DIR / 'drivers.csv')
    races = pd.read_csv(DATA_DIR / 'races.csv')
    constructors = pd.read_csv(DATA_DIR / 'constructors.csv')

    df = results.merge(races[['raceId', 'year', 'circuitId']], on='raceId', how='left')
    df = df.merge(drivers[['driverId', 'nationality']], on='driverId', how='left')
    df = df.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')

    for col in ['position', 'grid', 'points', 'laps', 'fastestLap', 'rank', 'fastestLapSpeed', 'statusId']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Target: podium finish (position 1-3). '\N'/DNF -> 0
    df[TARGET] = (df['position'] <= 3).astype(int)

    df_clean = df[FEATURES_NUM + [TARGET]].copy()

    # Encode categoricals (fit on full data, save encoders for inference)
    le_nat = LabelEncoder()
    le_con = LabelEncoder()
    nat = df.loc[df_clean.dropna().index, 'nationality'].fillna('Unknown')
    con = df.loc[df_clean.dropna().index, 'name'].fillna('Unknown')
    df_clean = df_clean.dropna()
    df_clean['nationality_encoded'] = le_nat.fit_transform(nat)
    df_clean['constructor_encoded'] = le_con.fit_transform(con)

    features_final = FEATURES_NUM + ['nationality_encoded', 'constructor_encoded']
    X = df_clean[features_final]
    y = df_clean[TARGET]
    return X, y, features_final, le_nat, le_con


def get_models(y_train):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / max(pos, 1)
    return {
        'random_forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42, class_weight='balanced', n_jobs=-1),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        'xgboost': XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
            eval_metric='logloss', scale_pos_weight=scale, n_jobs=-1),
    }


def main():
    print("Loading data...")
    X, y, features, le_nat, le_con = load_data()
    print(f"Samples: {len(X)} | Podiums: {int(y.sum())} ({y.mean()*100:.1f}%) | Features: {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # Save encoders + feature list once (shared by all models)
    joblib.dump({'features': features, 'le_nat': le_nat, 'le_con': le_con},
                MODEL_DIR / 'preprocessors.joblib')

    models = get_models(y_train)
    summary = {}

    for name, model in models.items():
        print(f"{'='*60}\nTraining {name}\n{'='*60}")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'time': train_time,
        }
        summary[name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | "
              f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
              f"AUC: {metrics['roc_auc']:.4f} | Time: {train_time:.2f}s")
        print(classification_report(y_test, y_pred, zero_division=0))

        out = MODEL_DIR / f'f1_model_{name}.joblib'
        joblib.dump(model, out)
        print(f"Saved -> {out}\n")

    print(f"{'='*60}\nCOMPARISON (sorted by F1)\n{'='*60}")
    df = pd.DataFrame(summary).T.sort_values('f1', ascending=False)
    print(df.to_string(float_format='%.4f'))
    best = df.index[0]
    print(f"\nBest model: {best} (F1={df.loc[best, 'f1']:.4f})")
    print(f"Models in: {MODEL_DIR.resolve()}")


if __name__ == '__main__':
    main()
