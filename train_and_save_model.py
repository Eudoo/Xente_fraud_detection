"""
Script d'entraînement et sauvegarde du modèle
Utilisé pour le retraining automatique
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def feature_engineering(df):
    """Création de nouvelles features"""
    df = df.copy()
    
    # Conversion de TransactionStartTime en datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extraction de features temporelles
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df['Month'] = df['TransactionStartTime'].dt.month
    df['DayOfMonth'] = df['TransactionStartTime'].dt.day
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNightTime'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
    
    # Ratio Amount/Value
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1)
    
    # Log transformation
    df['Log_Amount'] = np.log1p(df['Amount'].abs())
    df['Log_Value'] = np.log1p(df['Value'])
    
    # Encodage des IDs catégoriels
    id_columns = ['BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 
                  'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
    
    encoders = {}
    for col in id_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders


def train_model(data_path='xente-fraud-detection/training.csv', 
                model_dir='models',
                test_size=0.2,
                random_state=42):
    """
    Entraîne le modèle et sauvegarde tous les artefacts
    """
    logger.info("="*80)
    logger.info("DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE")
    logger.info("="*80)
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs(model_dir, exist_ok=True)
    
    # Chargement des données
    logger.info(f"Chargement des données depuis {data_path}")
    train_df = pd.read_csv(data_path)
    logger.info(f"Données chargées: {train_df.shape}")
    
    # Feature engineering
    logger.info("Application du feature engineering...")
    train_df, encoders = feature_engineering(train_df)
    logger.info(f"Feature engineering terminé. Nouvelles dimensions: {train_df.shape}")
    
    # Sélection des features
    numeric_features = ['Amount', 'Value', 'PricingStrategy', 
                       'Hour', 'DayOfWeek', 'Month', 'DayOfMonth',
                       'IsWeekend', 'IsNightTime', 'IsBusinessHours',
                       'Amount_Value_Ratio', 'Log_Amount', 'Log_Value',
                       'BatchId_encoded', 'AccountId_encoded', 'SubscriptionId_encoded',
                       'CustomerId_encoded', 'ProviderId_encoded', 'ProductId_encoded',
                       'ProductCategory_encoded', 'ChannelId_encoded']
    
    # Vérifier que toutes les features existent
    numeric_features = [f for f in numeric_features if f in train_df.columns]
    logger.info(f"Nombre de features utilisées: {len(numeric_features)}")
    
    # Séparation des features et de la cible
    X = train_df[numeric_features]
    y = train_df['FraudResult']
    
    # Division en ensembles d'entraînement et de validation
    logger.info("Division des données...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Taille train: {X_train.shape}, Taille validation: {X_val.shape}")
    
    # Normalisation des features
    logger.info("Normalisation des features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Entraînement du modèle
    logger.info("Entraînement du modèle Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("Entraînement terminé")
    
    # Évaluation
    logger.info("Évaluation du modèle...")
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"ROC-AUC: {roc_auc:.4f}")
    
    # Sauvegarde du modèle
    model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
    logger.info(f"Sauvegarde du modèle dans {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Sauvegarde du scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    logger.info(f"Sauvegarde du scaler dans {scaler_path}")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Sauvegarde des encoders
    encoders_path = os.path.join(model_dir, 'encoders.pkl')
    logger.info(f"Sauvegarde des encoders dans {encoders_path}")
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    
    # Sauvegarde des métadonnées
    metadata = {
        'training_date': datetime.now().isoformat(),
        'data_path': data_path,
        'num_samples': len(train_df),
        'num_features': len(numeric_features),
        'test_size': test_size,
        'random_state': random_state,
        'metrics': {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'feature_names': numeric_features
    }
    
    metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
    logger.info(f"Sauvegarde des métadonnées dans {metadata_path}")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    logger.info("="*80)
    logger.info("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    logger.info("="*80)
    
    return {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'metadata': metadata
    }


if __name__ == '__main__':
    # Entraîner et sauvegarder le modèle
    result = train_model()
    print("\n✅ Modèle entraîné et sauvegardé avec succès!")
    print(f"📊 Métriques:")
    print(f"   - Accuracy: {result['metadata']['metrics']['accuracy']:.4f}")
    print(f"   - F1-Score: {result['metadata']['metrics']['f1_score']:.4f}")
    print(f"   - ROC-AUC: {result['metadata']['metrics']['roc_auc']:.4f}")
