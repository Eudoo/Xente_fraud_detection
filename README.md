# 🔍 TP - Détection de Fraude Xente

## 📋 Description du Projet

Ce projet réalise une analyse complète de détection de fraude sur le dataset Xente, de A à Z. Il comprend :
- Analyse exploratoire des données (EDA)
- Feature engineering avancé
- Entraînement de plusieurs modèles de machine learning
- Visualisations détaillées
- Génération de prédictions

## 📁 Structure du Projet

```
Data science Projet/
│
├── xente-fraud-detection/          # Dataset
│   ├── training.csv                # Données d'entraînement
│   ├── test.csv                    # Données de test
│   ├── sample_submission.csv       # Exemple de soumission
│   └── Xente_Variable_Definitions.csv
│
├── fraud_detection_analysis.py     # Script principal d'analyse
├── visualizations/                 # Graphiques générés (9 visualisations)
├── submission.csv                  # Prédictions finales
├── rapport_final.txt              # Rapport détaillé
└── README.md                       # Ce fichier
```

## 🚀 Installation et Exécution

### Prérequis

Installez les bibliothèques nécessaires :

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Exécution

Lancez le script principal :

```bash
python fraud_detection_analysis.py
```

Le script va :
1. ✅ Charger et analyser les données
2. ✅ Créer des visualisations (9 graphiques)
3. ✅ Effectuer le feature engineering
4. ✅ Entraîner 3 modèles de ML
5. ✅ Comparer les performances
6. ✅ Générer les prédictions (submission.csv)
7. ✅ Créer un rapport final

## 📊 Étapes de l'Analyse

### Étape 1 : Chargement des Données
- Import des datasets training.csv et test.csv
- Analyse des dimensions et types de données
- Vérification de la qualité des données

### Étape 2 : Analyse Exploratoire (EDA)
- Statistiques descriptives
- Distribution de la variable cible (FraudResult)
- Détection des valeurs manquantes et doublons
- Analyse des corrélations

### Étape 3 : Visualisations
9 graphiques générés :
1. **Distribution Fraude vs Non-Fraude** - Vue d'ensemble du déséquilibre
2. **Distribution des Montants** - Comparaison par type de transaction
3. **Fraude par Catégorie de Produit** - Identification des catégories à risque
4. **Fraude par Canal** - Analyse des canaux de transaction
5. **Matrice de Corrélation** - Relations entre variables
6. **Comparaison des Modèles** - Performance des 3 modèles
7. **Courbes ROC** - Capacité discriminante
8. **Matrices de Confusion** - Erreurs de classification
9. **Feature Importance** - Variables les plus prédictives

### Étape 4 : Feature Engineering
Création de nouvelles variables :
- **Temporelles** : Hour, DayOfWeek, Month, IsWeekend, IsNightTime, IsBusinessHours
- **Ratios** : Amount_Value_Ratio
- **Transformations** : Log_Amount, Log_Value
- **Statistiques** : Account_Transaction_Count, Account_Avg_Amount, Amount_Deviation

### Étape 5 : Préparation des Données
- Encodage des variables catégorielles
- Division train/validation (80/20)
- Normalisation des features (StandardScaler)

### Étape 6 : Entraînement des Modèles

Trois modèles testés :
1. **Logistic Regression** - Modèle linéaire de base
2. **Random Forest** - Ensemble de décision trees
3. **Gradient Boosting** - Boosting séquentiel

### Étape 7 : Évaluation et Comparaison

Métriques utilisées :
- **Accuracy** - Taux de bonnes prédictions
- **F1-Score** - Équilibre précision/rappel
- **ROC-AUC** - Aire sous la courbe ROC

### Étape 8 : Visualisation des Performances
- Graphiques comparatifs des métriques
- Courbes ROC pour tous les modèles
- Matrices de confusion
- Importance des features

### Étape 9 : Prédictions
- Sélection du meilleur modèle (basé sur ROC-AUC)
- Génération des prédictions sur test.csv
- Création du fichier submission.csv

### Étape 10 : Rapport Final
- Résumé complet de l'analyse
- Statistiques clés
- Performances des modèles
- Liste des fichiers générés

## 📈 Résultats Attendus

Le script génère automatiquement :
- ✅ **9 visualisations** dans le dossier `visualizations/`
- ✅ **submission.csv** avec les prédictions
- ✅ **rapport_final.txt** avec le résumé complet

## 🎯 Variables du Dataset

| Variable | Description |
|----------|-------------|
| TransactionId | Identifiant unique de transaction |
| BatchId | Numéro de lot de transactions |
| AccountId | Identifiant du client |
| SubscriptionId | Identifiant de l'abonnement |
| CustomerId | Identifiant client attaché au compte |
| CurrencyCode | Code devise du pays |
| CountryCode | Code géographique du pays |
| ProviderId | Fournisseur de l'article acheté |
| ProductId | Nom de l'article acheté |
| ProductCategory | Catégorie de produit |
| ChannelId | Canal utilisé (web, Android, iOS, etc.) |
| Amount | Valeur de la transaction |
| Value | Valeur absolue du montant |
| TransactionStartTime | Heure de début de transaction |
| PricingStrategy | Structure de prix Xente |
| **FraudResult** | **Statut de fraude (0=Non, 1=Oui)** ⭐ |

## 🔧 Technologies Utilisées

- **Python 3.x**
- **pandas** - Manipulation de données
- **numpy** - Calculs numériques
- **matplotlib** - Visualisations
- **seaborn** - Visualisations statistiques
- **scikit-learn** - Machine Learning

## 📝 Notes Importantes

- Le dataset présente un **déséquilibre de classes** (peu de fraudes)
- Les modèles utilisent **class_weight='balanced'** pour compenser
- La métrique principale est **ROC-AUC** (adaptée aux classes déséquilibrées)
- Le feature engineering améliore significativement les performances

## 🎓 Apprentissages Clés

1. **Importance du Feature Engineering** - Les features temporelles sont cruciales
2. **Gestion du Déséquilibre** - Utilisation de class_weight et métriques adaptées
3. **Comparaison de Modèles** - Random Forest et Gradient Boosting performent mieux
4. **Visualisation** - Essentielle pour comprendre les patterns de fraude

## 📧 Support

Pour toute question sur ce TP, consultez :
- Le fichier `rapport_final.txt` pour les résultats détaillés
- Les visualisations dans `visualizations/` pour l'analyse graphique
- Le code commenté dans `fraud_detection_analysis.py`


