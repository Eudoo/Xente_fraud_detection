"""
Script de retraining automatique
Peut être exécuté par un cron job ou un scheduler
"""

import os
import sys
import logging
from datetime import datetime
import shutil
from train_and_save_model import train_model

# Configuration du logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'retraining_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def backup_current_model(model_dir='models', backup_dir='models_backup'):
    """
    Sauvegarde le modèle actuel avant le retraining
    """
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f'backup_{timestamp}')
        
        if os.path.exists(model_dir):
            shutil.copytree(model_dir, backup_path)
            logger.info(f"Modèle sauvegardé dans {backup_path}")
            return backup_path
        else:
            logger.warning("Aucun modèle existant à sauvegarder")
            return None
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
        return None


def cleanup_old_backups(backup_dir='models_backup', keep_last_n=5):
    """
    Supprime les anciennes sauvegardes pour économiser l'espace
    """
    try:
        if not os.path.exists(backup_dir):
            return
        
        backups = sorted([
            os.path.join(backup_dir, d) 
            for d in os.listdir(backup_dir) 
            if os.path.isdir(os.path.join(backup_dir, d))
        ], key=os.path.getmtime, reverse=True)
        
        # Garder seulement les N dernières sauvegardes
        for backup in backups[keep_last_n:]:
            shutil.rmtree(backup)
            logger.info(f"Sauvegarde supprimée: {backup}")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des sauvegardes: {e}")


def send_notification(status, message):
    """
    Envoie une notification (email, Slack, etc.)
    À personnaliser selon vos besoins
    """
    # Exemple: écrire dans un fichier de notifications
    notification_file = 'logs/notifications.log'
    with open(notification_file, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] {status}: {message}\n")
    
    logger.info(f"Notification: {status} - {message}")


def main():
    """
    Fonction principale de retraining
    """
    logger.info("="*80)
    logger.info("DÉBUT DU PROCESSUS DE RETRAINING AUTOMATIQUE")
    logger.info("="*80)
    
    try:
        # 1. Sauvegarder le modèle actuel
        logger.info("Étape 1: Sauvegarde du modèle actuel")
        backup_path = backup_current_model()
        
        # 2. Entraîner le nouveau modèle
        logger.info("Étape 2: Entraînement du nouveau modèle")
        result = train_model(
            data_path='xente-fraud-detection/training.csv',
            model_dir='models'
        )
        
        # 3. Vérifier les performances
        logger.info("Étape 3: Vérification des performances")
        metrics = result['metadata']['metrics']
        
        # Seuil minimum de performance
        min_roc_auc = 0.85
        
        if metrics['roc_auc'] < min_roc_auc:
            logger.warning(f"Performance insuffisante (ROC-AUC: {metrics['roc_auc']:.4f} < {min_roc_auc})")
            
            # Restaurer l'ancien modèle
            if backup_path:
                logger.info("Restauration de l'ancien modèle")
                if os.path.exists('models'):
                    shutil.rmtree('models')
                shutil.copytree(backup_path, 'models')
            
            send_notification('WARNING', f"Retraining échoué - Performance insuffisante: {metrics['roc_auc']:.4f}")
            return False
        
        # 4. Nettoyer les anciennes sauvegardes
        logger.info("Étape 4: Nettoyage des anciennes sauvegardes")
        cleanup_old_backups(keep_last_n=5)
        
        # 5. Notification de succès
        logger.info("="*80)
        logger.info("RETRAINING TERMINÉ AVEC SUCCÈS")
        logger.info(f"Nouvelles métriques:")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info("="*80)
        
        send_notification('SUCCESS', f"Retraining réussi - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du retraining: {e}", exc_info=True)
        send_notification('ERROR', f"Erreur lors du retraining: {str(e)}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
