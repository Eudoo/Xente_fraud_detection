#!/bin/bash

# Script de planification avec cron
# Ce script configure un cron job pour le retraining automatique

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Configuration du Retraining Automatique${NC}"
echo -e "${BLUE}========================================${NC}"

# Chemin du script Python
SCRIPT_PATH="$(pwd)/automated_retraining.py"
PYTHON_PATH=$(which python)

# Vérifier que le script existe
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Erreur: Le script automated_retraining.py n'existe pas"
    exit 1
fi

echo -e "\n${GREEN}Script trouvé:${NC} $SCRIPT_PATH"
echo -e "${GREEN}Python:${NC} $PYTHON_PATH"

# Créer le cron job
# Par défaut: tous les jours à 2h du matin
CRON_SCHEDULE="0 2 * * *"

# Créer une entrée cron
CRON_COMMAND="$CRON_SCHEDULE cd $(pwd) && $PYTHON_PATH $SCRIPT_PATH >> logs/cron_retraining.log 2>&1"

echo -e "\n${BLUE}Configuration du cron job:${NC}"
echo "  Planification: Tous les jours à 2h00"
echo "  Commande: $CRON_COMMAND"

# Ajouter au crontab
(crontab -l 2>/dev/null; echo "$CRON_COMMAND") | crontab -

echo -e "\n${GREEN}✓ Cron job configuré avec succès!${NC}"
echo -e "\nPour vérifier les cron jobs actuels:"
echo "  crontab -l"
echo -e "\nPour voir les logs:"
echo "  tail -f logs/cron_retraining.log"

# Créer le dossier logs s'il n'existe pas
mkdir -p logs

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Configuration terminée!${NC}"
echo -e "${BLUE}========================================${NC}"
