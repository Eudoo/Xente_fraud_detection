"""
API Flask pour la Détection de Fraude Xente
Déploiement du modèle en production
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Configuration
MODEL_PATH = 'models/fraud_detection_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODERS_PATH = 'models/encoders.pkl'

# Chargement du modèle et des transformateurs
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    logger.info("Modèle et transformateurs chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    model = None
    scaler = None
    encoders = None

# Statistiques de l'API
api_stats = {
    'total_predictions': 0,
    'fraud_detected': 0,
    'start_time': datetime.now().isoformat()
}


def feature_engineering(data):
    """
    Applique le feature engineering sur les données d'entrée
    """
    df = pd.DataFrame([data])
    
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
    
    for col in id_columns:
        if col in df.columns and col in encoders:
            try:
                df[f'{col}_encoded'] = encoders[col].transform(df[col].astype(str))
            except:
                # Si la valeur n'existe pas dans l'encodeur, utiliser 0
                df[f'{col}_encoded'] = 0
    
    return df


def prepare_features(df):
    """
    Prépare les features pour la prédiction
    """
    feature_columns = ['Amount', 'Value', 'PricingStrategy', 
                      'Hour', 'DayOfWeek', 'Month', 'DayOfMonth',
                      'IsWeekend', 'IsNightTime', 'IsBusinessHours',
                      'Amount_Value_Ratio', 'Log_Amount', 'Log_Value',
                      'BatchId_encoded', 'AccountId_encoded', 'SubscriptionId_encoded',
                      'CustomerId_encoded', 'ProviderId_encoded', 'ProductId_encoded',
                      'ProductCategory_encoded', 'ChannelId_encoded']
    
    # Vérifier que toutes les colonnes existent
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_columns]


@app.route('/')
def home():
    """
    Page d'accueil de l'API
    """
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint de vérification de santé de l'API
    """
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoders_loaded': encoders is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)


@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Retourne les statistiques de l'API
    """
    stats = api_stats.copy()
    stats['uptime'] = str(datetime.now() - datetime.fromisoformat(stats['start_time']))
    if stats['total_predictions'] > 0:
        stats['fraud_rate'] = (stats['fraud_detected'] / stats['total_predictions']) * 100
    else:
        stats['fraud_rate'] = 0
    return jsonify(stats)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    Accepte une transaction et retourne la probabilité de fraude
    """
    try:
        # Vérifier que le modèle est chargé
        if model is None:
            return jsonify({
                'error': 'Modèle non chargé',
                'message': 'Le modèle n\'a pas pu être chargé au démarrage'
            }), 500
        
        # Récupérer les données de la requête
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'Données manquantes',
                'message': 'Veuillez fournir les données de transaction'
            }), 400
        
        # Validation des champs requis
        required_fields = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId',
                          'CustomerId', 'ProviderId', 'ProductId', 'ProductCategory',
                          'ChannelId', 'Amount', 'Value', 'TransactionStartTime',
                          'PricingStrategy']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Champs manquants',
                'missing_fields': missing_fields
            }), 400
        
        # Feature engineering
        df = feature_engineering(data)
        
        # Préparation des features
        X = prepare_features(df)
        
        # Prédiction
        fraud_probability = model.predict_proba(X)[0][1]
        fraud_prediction = int(fraud_probability > 0.5)
        
        # Mise à jour des statistiques
        api_stats['total_predictions'] += 1
        if fraud_prediction == 1:
            api_stats['fraud_detected'] += 1
        
        # Résultat
        result = {
            'transaction_id': data['TransactionId'],
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(fraud_prediction),
            'risk_level': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prédiction effectuée pour {data['TransactionId']}: {fraud_probability:.4f}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({
            'error': 'Erreur de prédiction',
            'message': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Endpoint de prédiction par lot
    Accepte plusieurs transactions et retourne les probabilités de fraude
    """
    try:
        if model is None:
            return jsonify({
                'error': 'Modèle non chargé'
            }), 500
        
        # Récupérer les données
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({
                'error': 'Format de données invalide',
                'message': 'Veuillez fournir un tableau "transactions"'
            }), 400
        
        transactions = data['transactions']
        results = []
        
        for transaction in transactions:
            try:
                # Feature engineering
                df = feature_engineering(transaction)
                
                # Préparation des features
                X = prepare_features(df)
                
                # Prédiction
                fraud_probability = model.predict_proba(X)[0][1]
                fraud_prediction = int(fraud_probability > 0.5)
                
                # Mise à jour des statistiques
                api_stats['total_predictions'] += 1
                if fraud_prediction == 1:
                    api_stats['fraud_detected'] += 1
                
                results.append({
                    'transaction_id': transaction['TransactionId'],
                    'fraud_probability': float(fraud_probability),
                    'is_fraud': bool(fraud_prediction),
                    'risk_level': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.3 else 'LOW'
                })
            except Exception as e:
                results.append({
                    'transaction_id': transaction.get('TransactionId', 'unknown'),
                    'error': str(e)
                })
        
        logger.info(f"Prédiction batch effectuée pour {len(transactions)} transactions")
        
        return jsonify({
            'total_transactions': len(transactions),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction batch: {e}")
        return jsonify({
            'error': 'Erreur de prédiction batch',
            'message': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Retourne les informations sur le modèle
    """
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    info = {
        'model_type': type(model).__name__,
        'model_path': MODEL_PATH,
        'features_count': 21,
        'last_updated': datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat() if os.path.exists(MODEL_PATH) else None
    }
    
    return jsonify(info)


if __name__ == '__main__':
    # Créer le dossier models s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    
    # Lancer l'application
    logger.info("Démarrage de l'API de détection de fraude")
    app.run(host='0.0.0.0', port=5000, debug=True)
