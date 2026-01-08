# Dockerfile pour l'API de Détection de Fraude Xente
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask flask-cors gunicorn

# Copier tous les fichiers de l'application
COPY api_fraud_detection.py .
COPY train_and_save_model.py .
COPY automated_retraining.py .
COPY templates/ templates/
COPY xente-fraud-detection/ xente-fraud-detection/

# Créer les dossiers nécessaires
RUN mkdir -p models logs models_backup

# Entraîner le modèle initial
RUN python train_and_save_model.py

# Exposer le port 5000
EXPOSE 5000

# Variable d'environnement
ENV FLASK_APP=api_fraud_detection.py
ENV PYTHONUNBUFFERED=1

# Commande de démarrage avec Gunicorn (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "api_fraud_detection:app"]
