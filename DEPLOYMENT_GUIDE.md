# 🚀 GUIDE DE DÉPLOIEMENT - API DÉTECTION DE FRAUDE XENTE

## 📋 Table des Matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Installation Locale](#installation-locale)
4. [Déploiement avec Docker](#déploiement-avec-docker)
5. [Automatisation du Retraining](#automatisation-du-retraining)
6. [Utilisation de l'API](#utilisation-de-lapi)
7. [Monitoring et Maintenance](#monitoring-et-maintenance)

---

## 🎯 Vue d'ensemble

Cette solution de déploiement comprend :

- **API REST Flask** pour servir le modèle en production
- **Application Web** interactive pour tester l'API
- **Script de retraining automatique** avec sauvegarde et validation
- **Conteneurisation Docker** pour un déploiement facile
- **Planification automatique** (cron/Task Scheduler)

### Architecture

```
┌─────────────────┐
│   Client Web    │
│  (Browser/App)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   API Flask     │
│  (Port 5000)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Modèle ML      │
│ (Random Forest) │
└─────────────────┘
         ▲
         │
┌────────┴────────┐
│   Retraining    │
│   Automatique   │
└─────────────────┘
```

---

## 📦 Prérequis

### Logiciels Requis

- **Python 3.9+**
- **pip** (gestionnaire de paquets Python)
- **Docker** (optionnel, pour la conteneurisation)
- **Docker Compose** (optionnel)

### Vérification

```bash
python --version
pip --version
docker --version
docker-compose --version
```

---

## 💻 Installation Locale

### Étape 1: Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 2: Entraîner le modèle initial

```bash
python train_and_save_model.py
```

Cela va créer le dossier `models/` avec :
- `fraud_detection_model.pkl` - Le modèle Random Forest
- `scaler.pkl` - Le StandardScaler
- `encoders.pkl` - Les LabelEncoders
- `model_metadata.pkl` - Les métadonnées

### Étape 3: Lancer l'API

```bash
python api_fraud_detection.py
```

L'API sera accessible sur : **http://localhost:5000**

### Étape 4: Tester l'API

Ouvrez votre navigateur et allez sur : **http://localhost:5000**

Vous verrez l'interface web interactive pour tester les prédictions.

---

## 🐳 Déploiement avec Docker

### Option 1: Docker Simple

#### 1. Construire l'image

```bash
docker build -t fraud-detection-api .
```

#### 2. Lancer le conteneur

```bash
docker run -d \
  --name fraud-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fraud-detection-api
```

#### 3. Vérifier le statut

```bash
docker ps
docker logs fraud-api
```

### Option 2: Docker Compose (Recommandé)

#### 1. Lancer tous les services

```bash
docker-compose up -d
```

Cela lance :
- **fraud-detection-api** : L'API sur le port 5000
- **retraining-scheduler** : Le scheduler de retraining (toutes les 24h)

#### 2. Vérifier les services

```bash
docker-compose ps
docker-compose logs -f fraud-detection-api
```

#### 3. Arrêter les services

```bash
docker-compose down
```

### Commandes Docker Utiles

```bash
# Voir les logs en temps réel
docker-compose logs -f

# Redémarrer un service
docker-compose restart fraud-detection-api

# Reconstruire les images
docker-compose build --no-cache

# Nettoyer tout
docker-compose down -v
```

---

## ⚙️ Automatisation du Retraining

Le retraining automatique permet de mettre à jour le modèle régulièrement avec de nouvelles données.

### Fonctionnalités

✅ Sauvegarde automatique du modèle actuel  
✅ Validation des performances du nouveau modèle  
✅ Rollback automatique si les performances sont insuffisantes  
✅ Nettoyage des anciennes sauvegardes  
✅ Notifications de succès/échec  

### Option 1: Planification Windows (Task Scheduler)

```powershell
# Exécuter le script PowerShell en tant qu'administrateur
.\setup_scheduled_task.ps1
```

Cela crée une tâche planifiée qui s'exécute **tous les jours à 2h00**.

#### Vérifier la tâche

1. Ouvrir le **Planificateur de tâches** Windows
2. Rechercher : `FraudDetection_AutoRetraining`
3. Voir l'historique et les logs

### Option 2: Planification Linux/Mac (Cron)

```bash
# Rendre le script exécutable
chmod +x setup_cron.sh

# Exécuter le script
./setup_cron.sh
```

#### Vérifier le cron job

```bash
# Voir les cron jobs
crontab -l

# Voir les logs
tail -f logs/cron_retraining.log
```

### Option 3: Docker Compose (Automatique)

Si vous utilisez `docker-compose.yml`, le retraining est déjà planifié automatiquement toutes les 24 heures.

### Exécution Manuelle

```bash
python automated_retraining.py
```

---

## 🔌 Utilisation de l'API

### Endpoints Disponibles

#### 1. Health Check

```bash
GET /health
```

**Exemple:**
```bash
curl http://localhost:5000/health
```

**Réponse:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "encoders_loaded": true,
  "timestamp": "2026-01-08T01:30:00"
}
```

#### 2. Statistiques

```bash
GET /stats
```

**Exemple:**
```bash
curl http://localhost:5000/stats
```

**Réponse:**
```json
{
  "total_predictions": 1523,
  "fraud_detected": 12,
  "fraud_rate": 0.79,
  "uptime": "2 days, 5:30:15",
  "start_time": "2026-01-05T20:00:00"
}
```

#### 3. Prédiction Simple

```bash
POST /predict
Content-Type: application/json
```

**Exemple:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionId": "TransactionId_12345",
    "BatchId": "BatchId_12345",
    "AccountId": "AccountId_12345",
    "SubscriptionId": "SubscriptionId_12345",
    "CustomerId": "CustomerId_12345",
    "ProviderId": "ProviderId_123",
    "ProductId": "ProductId_123",
    "ProductCategory": "airtime",
    "ChannelId": "ChannelId_1",
    "Amount": 1000,
    "Value": 1000,
    "TransactionStartTime": "2026-01-08T10:30:00",
    "PricingStrategy": 2
  }'
```

**Réponse:**
```json
{
  "transaction_id": "TransactionId_12345",
  "fraud_probability": 0.0234,
  "is_fraud": false,
  "risk_level": "LOW",
  "timestamp": "2026-01-08T01:30:00"
}
```

#### 4. Prédiction par Lot

```bash
POST /predict_batch
Content-Type: application/json
```

**Exemple:**
```bash
curl -X POST http://localhost:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      { /* transaction 1 */ },
      { /* transaction 2 */ }
    ]
  }'
```

#### 5. Informations sur le Modèle

```bash
GET /model_info
```

**Exemple:**
```bash
curl http://localhost:5000/model_info
```

**Réponse:**
```json
{
  "model_type": "RandomForestClassifier",
  "model_path": "models/fraud_detection_model.pkl",
  "features_count": 21,
  "last_updated": "2026-01-08T00:00:00"
}
```

### Interface Web

Accédez à **http://localhost:5000** pour utiliser l'interface web interactive.

Fonctionnalités :
- ✅ Formulaire de test de transaction
- ✅ Visualisation des résultats en temps réel
- ✅ Statistiques de l'API
- ✅ Documentation des endpoints

---

## 📊 Monitoring et Maintenance

### Logs

Tous les logs sont stockés dans le dossier `logs/` :

```
logs/
├── api_logs.log                    # Logs de l'API
├── training_logs.log               # Logs d'entraînement
├── retraining_YYYYMMDD_HHMMSS.log # Logs de retraining
├── cron_retraining.log             # Logs du cron
└── notifications.log               # Notifications
```

### Consulter les Logs

```bash
# Logs de l'API en temps réel
tail -f logs/api_logs.log

# Derniers logs de retraining
ls -t logs/retraining_*.log | head -1 | xargs cat

# Notifications
cat logs/notifications.log
```

### Sauvegardes du Modèle

Les sauvegardes sont stockées dans `models_backup/` :

```
models_backup/
├── backup_20260108_020000/
├── backup_20260107_020000/
└── backup_20260106_020000/
```

Les 5 dernières sauvegardes sont conservées automatiquement.

### Métriques à Surveiller

1. **Performance du Modèle**
   - ROC-AUC > 0.85 (seuil minimum)
   - F1-Score
   - Accuracy

2. **API**
   - Temps de réponse
   - Taux d'erreur
   - Nombre de prédictions

3. **Système**
   - Utilisation CPU/RAM
   - Espace disque
   - Uptime

---

## 🔧 Configuration Avancée

### Variables d'Environnement

Créez un fichier `.env` :

```env
FLASK_ENV=production
MODEL_PATH=models/fraud_detection_model.pkl
SCALER_PATH=models/scaler.pkl
ENCODERS_PATH=models/encoders.pkl
MIN_ROC_AUC=0.85
RETRAINING_INTERVAL=86400  # 24 heures en secondes
```

### Production avec Gunicorn

```bash
gunicorn --bind 0.0.0.0:5000 \
         --workers 4 \
         --timeout 120 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         api_fraud_detection:app
```

### Reverse Proxy avec Nginx

```nginx
server {
    listen 80;
    server_name fraud-detection.example.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 🚨 Dépannage

### Problème: Le modèle ne se charge pas

**Solution:**
```bash
# Vérifier que le modèle existe
ls -la models/

# Réentraîner le modèle
python train_and_save_model.py
```

### Problème: Erreur de prédiction

**Solution:**
```bash
# Vérifier les logs
tail -f logs/api_logs.log

# Tester avec curl
curl http://localhost:5000/health
```

### Problème: Docker ne démarre pas

**Solution:**
```bash
# Voir les logs
docker-compose logs

# Reconstruire
docker-compose build --no-cache
docker-compose up -d
```

---

## 📚 Ressources

- **Documentation Flask**: https://flask.palletsprojects.com/
- **Documentation Docker**: https://docs.docker.com/
- **Scikit-learn**: https://scikit-learn.org/

---

## ✅ Checklist de Déploiement

- [ ] Dépendances installées (`pip install -r requirements.txt`)
- [ ] Modèle entraîné (`python train_and_save_model.py`)
- [ ] API testée localement (`python api_fraud_detection.py`)
- [ ] Docker configuré (optionnel)
- [ ] Retraining automatique planifié
- [ ] Logs configurés et surveillés
- [ ] Sauvegardes vérifiées
- [ ] Documentation lue et comprise

---

**🎉 Félicitations ! Votre API de détection de fraude est déployée et opérationnelle !**
