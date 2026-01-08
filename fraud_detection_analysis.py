"""
TP - Détection de Fraude Xente - Version Optimisée
Analyse complète de A à Z
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve,
    f1_score,
    accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Configuration pour les graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("TP - DÉTECTION DE FRAUDE XENTE - ANALYSE COMPLÈTE")
print("="*80)

# ============================================================================
# ÉTAPE 1: CHARGEMENT DES DONNÉES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 1: CHARGEMENT DES DONNÉES")
print("="*80)

# Chargement des données
train_df = pd.read_csv('xente-fraud-detection/training.csv')
test_df = pd.read_csv('xente-fraud-detection/test.csv')
variable_def = pd.read_csv('xente-fraud-detection/Xente_Variable_Definitions.csv')

print(f"\nDimensions du dataset d'entraînement: {train_df.shape}")
print(f"Dimensions du dataset de test: {test_df.shape}")

print("\n--- Aperçu des données d'entraînement ---")
print(train_df.head())

# ============================================================================
# ÉTAPE 2: ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 2: ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
print("="*80)

# Statistiques descriptives
print("\n--- Statistiques descriptives ---")
print(train_df.describe())

# Distribution de la variable cible
print("\n--- Distribution de la variable cible (FraudResult) ---")
fraud_counts = train_df['FraudResult'].value_counts()
print(fraud_counts)
print(f"\nPourcentage de fraudes: {fraud_counts[1] / len(train_df) * 100:.2f}%")
print(f"Pourcentage de non-fraudes: {fraud_counts[0] / len(train_df) * 100:.2f}%")

# Vérification des valeurs manquantes
print("\n--- Valeurs manquantes ---")
missing_values = train_df.isnull().sum()
print(missing_values[missing_values > 0])
if missing_values.sum() == 0:
    print("Aucune valeur manquante détectée!")

# ============================================================================
# ÉTAPE 3: VISUALISATIONS
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 3: CRÉATION DES VISUALISATIONS")
print("="*80)

# Création d'un dossier pour les visualisations
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Distribution de la variable cible
plt.figure(figsize=(10, 6))
fraud_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Distribution des Transactions (Fraude vs Non-Fraude)', fontsize=16, fontweight='bold')
plt.xlabel('FraudResult (0=Non-Fraude, 1=Fraude)', fontsize=12)
plt.ylabel('Nombre de transactions', fontsize=12)
plt.xticks(rotation=0)
for i, v in enumerate(fraud_counts):
    plt.text(i, v + 1000, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/1_distribution_fraude.png', dpi=300)
print("✓ Graphique 1 sauvegardé: distribution_fraude.png")
plt.close()

# 2. Distribution des montants par type de transaction
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
train_df[train_df['FraudResult'] == 0]['Amount'].hist(bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution des montants - Non-Fraude', fontweight='bold')
plt.xlabel('Montant')
plt.ylabel('Fréquence')

plt.subplot(1, 2, 2)
train_df[train_df['FraudResult'] == 1]['Amount'].hist(bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution des montants - Fraude', fontweight='bold')
plt.xlabel('Montant')
plt.ylabel('Fréquence')
plt.tight_layout()
plt.savefig('visualizations/2_distribution_montants.png', dpi=300)
print("✓ Graphique 2 sauvegardé: distribution_montants.png")
plt.close()

# 3. Fraude par catégorie de produit
plt.figure(figsize=(14, 6))
fraud_by_category = train_df.groupby('ProductCategory')['FraudResult'].agg(['sum', 'count'])
fraud_by_category['fraud_rate'] = (fraud_by_category['sum'] / fraud_by_category['count'] * 100)
fraud_by_category = fraud_by_category.sort_values('fraud_rate', ascending=False)
fraud_by_category['fraud_rate'].plot(kind='bar', color='coral')
plt.title('Taux de Fraude par Catégorie de Produit', fontsize=16, fontweight='bold')
plt.xlabel('Catégorie de Produit', fontsize=12)
plt.ylabel('Taux de Fraude (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/3_fraude_par_categorie.png', dpi=300)
print("✓ Graphique 3 sauvegardé: fraude_par_categorie.png")
plt.close()

# 4. Fraude par canal (ChannelId)
plt.figure(figsize=(12, 6))
fraud_by_channel = train_df.groupby('ChannelId')['FraudResult'].agg(['sum', 'count'])
fraud_by_channel['fraud_rate'] = (fraud_by_channel['sum'] / fraud_by_channel['count'] * 100)
fraud_by_channel = fraud_by_channel.sort_values('fraud_rate', ascending=False)
fraud_by_channel['fraud_rate'].plot(kind='bar', color='skyblue')
plt.title('Taux de Fraude par Canal', fontsize=16, fontweight='bold')
plt.xlabel('Canal ID', fontsize=12)
plt.ylabel('Taux de Fraude (%)', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('visualizations/4_fraude_par_canal.png', dpi=300)
print("✓ Graphique 4 sauvegardé: fraude_par_canal.png")
plt.close()

# 5. Matrice de corrélation
plt.figure(figsize=(12, 10))
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
correlation_matrix = train_df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Matrice de Corrélation des Variables Numériques', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/5_matrice_correlation.png', dpi=300)
print("✓ Graphique 5 sauvegardé: matrice_correlation.png")
plt.close()

# ============================================================================
# ÉTAPE 4: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 4: FEATURE ENGINEERING")
print("="*80)

def feature_engineering(df):
    """Création de nouvelles features"""
    df = df.copy()
    
    # Conversion de TransactionStartTime en datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extraction de features temporelles
    df['Hour'] = df['TransactionStartTime'].dt.hour
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df['Month'] = df['TransactionStartTime'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNightTime'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
    
    # Ratio Amount/Value
    df['Amount_Value_Ratio'] = df['Amount'] / (df['Value'] + 1)
    
    # Log transformation
    df['Log_Amount'] = np.log1p(df['Amount'].abs())
    df['Log_Value'] = np.log1p(df['Value'])
    
    return df

# Application du feature engineering
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

print("✓ Features temporelles créées")
print("✓ Features de ratio créées")
print("✓ Features de transformation logarithmique créées")

# ============================================================================
# ÉTAPE 5: PRÉPARATION DES DONNÉES POUR LE MODÈLE
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 5: PRÉPARATION DES DONNÉES POUR LE MODÈLE")
print("="*80)

# Sélection des features numériques uniquement pour simplifier
numeric_features = ['BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 
                   'ProviderId', 'ProductId', 'ChannelId', 'Amount', 'Value',
                   'PricingStrategy', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
                   'IsNightTime', 'Amount_Value_Ratio', 'Log_Amount', 'Log_Value']

# Vérifier que toutes les features existent
numeric_features = [f for f in numeric_features if f in train_df.columns]

print(f"\nNombre de features utilisées: {len(numeric_features)}")

# Séparation des features et de la cible
X = train_df[numeric_features]
y = train_df['FraudResult']

# Division en ensembles d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTaille de l'ensemble d'entraînement: {X_train.shape}")
print(f"Taille de l'ensemble de validation: {X_val.shape}")

# Normalisation des features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("✓ Normalisation des features terminée")

# ============================================================================
# ÉTAPE 6: ENTRAÎNEMENT DES MODÈLES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 6: ENTRAÎNEMENT DES MODÈLES")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Entraînement: {name} ---")
    
    # Entraînement
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Évaluation
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# ÉTAPE 7: COMPARAISON DES MODÈLES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 7: COMPARAISON DES MODÈLES")
print("="*80)

# Tableau comparatif
comparison_df = pd.DataFrame({
    'Modèle': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'F1-Score': [results[m]['f1_score'] for m in results.keys()],
    'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
})

print("\n", comparison_df.to_string(index=False))

# Meilleur modèle
best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Modèle']
print(f"\n🏆 Meilleur modèle: {best_model_name}")

# ============================================================================
# ÉTAPE 8: VISUALISATIONS DES PERFORMANCES
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 8: VISUALISATIONS DES PERFORMANCES")
print("="*80)

# 1. Comparaison des métriques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
colors = ['#3498db', '#e74c3c', '#2ecc71']

for idx, metric in enumerate(metrics):
    axes[idx].bar(comparison_df['Modèle'], comparison_df[metric], color=colors[idx], alpha=0.7)
    axes[idx].set_title(f'Comparaison - {metric}', fontweight='bold', fontsize=14)
    axes[idx].set_ylabel(metric, fontsize=12)
    axes[idx].set_xticklabels(comparison_df['Modèle'], rotation=45, ha='right')
    axes[idx].set_ylim([0, 1])
    
    for i, v in enumerate(comparison_df[metric]):
        axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/6_comparaison_modeles.png', dpi=300)
print("✓ Graphique 6 sauvegardé: comparaison_modeles.png")
plt.close()

# 2. Courbes ROC
plt.figure(figsize=(10, 8))
for name in results.keys():
    fpr, tpr, _ = roc_curve(y_val, results[name]['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
plt.xlabel('Taux de Faux Positifs', fontsize=12)
plt.ylabel('Taux de Vrais Positifs', fontsize=12)
plt.title('Courbes ROC - Comparaison des Modèles', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/7_courbes_roc.png', dpi=300)
print("✓ Graphique 7 sauvegardé: courbes_roc.png")
plt.close()

# 3. Matrices de confusion
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, name in enumerate(results.keys()):
    cm = confusion_matrix(y_val, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'Matrice de Confusion - {name}', fontweight='bold', fontsize=12)
    axes[idx].set_xlabel('Prédiction', fontsize=10)
    axes[idx].set_ylabel('Réalité', fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/8_matrices_confusion.png', dpi=300)
print("✓ Graphique 8 sauvegardé: matrices_confusion.png")
plt.close()

# 4. Feature Importance (pour Random Forest)
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': numeric_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'], color='teal')
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 15 Features les Plus Importantes (Random Forest)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/9_feature_importance.png', dpi=300)
    print("✓ Graphique 9 sauvegardé: feature_importance.png")
    plt.close()

# ============================================================================
# ÉTAPE 9: PRÉDICTIONS SUR LE TEST SET
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 9: PRÉDICTIONS SUR LE TEST SET")
print("="*80)

# Utilisation du meilleur modèle
best_model = results[best_model_name]['model']

# Préparation des données de test
X_test = test_df[numeric_features]

# Prédictions
if best_model_name == 'Logistic Regression':
    X_test_scaled = scaler.transform(X_test)
    test_predictions = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    test_predictions = best_model.predict_proba(X_test)[:, 1]

# Création du fichier de soumission
submission = pd.DataFrame({
    'TransactionId': test_df['TransactionId'],
    'FraudResult': test_predictions
})

submission.to_csv('submission.csv', index=False)
print(f"✓ Fichier de soumission créé: submission.csv")
print(f"  Nombre de prédictions: {len(submission)}")
print(f"\nAperçu des prédictions:")
print(submission.head(10))

# ============================================================================
# ÉTAPE 10: RAPPORT FINAL
# ============================================================================
print("\n" + "="*80)
print("ÉTAPE 10: RAPPORT FINAL")
print("="*80)

report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RAPPORT FINAL - DÉTECTION DE FRAUDE XENTE                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 STATISTIQUES DU DATASET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Nombre total de transactions (train): {len(train_df):,}
  • Nombre total de transactions (test): {len(test_df):,}
  • Nombre de features utilisées: {len(numeric_features)}
  • Taux de fraude dans le dataset: {fraud_counts[1] / len(train_df) * 100:.2f}%

🔧 FEATURE ENGINEERING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Features temporelles: Hour, DayOfWeek, Month, IsWeekend, IsNightTime
  • Features de ratio: Amount_Value_Ratio
  • Features de transformation: Log_Amount, Log_Value

🤖 MODÈLES TESTÉS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

for name in results.keys():
    report += f"""
  {name}:
    - Accuracy:  {results[name]['accuracy']:.4f}
    - F1-Score:  {results[name]['f1_score']:.4f}
    - ROC-AUC:   {results[name]['roc_auc']:.4f}
"""

report += f"""
🏆 MEILLEUR MODÈLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Modèle sélectionné: {best_model_name}
  ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}

📁 FICHIERS GÉNÉRÉS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • submission.csv - Prédictions pour le test set
  • visualizations/ - Dossier contenant 9 graphiques d'analyse
    ✓ 1_distribution_fraude.png
    ✓ 2_distribution_montants.png
    ✓ 3_fraude_par_categorie.png
    ✓ 4_fraude_par_canal.png
    ✓ 5_matrice_correlation.png
    ✓ 6_comparaison_modeles.png
    ✓ 7_courbes_roc.png
    ✓ 8_matrices_confusion.png
    ✓ 9_feature_importance.png

✅ ANALYSE TERMINÉE AVEC SUCCÈS!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

print(report)

# Sauvegarde du rapport
with open('rapport_final.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("\n✓ Rapport final sauvegardé: rapport_final.txt")
print("\n" + "="*80)
print("🎉 TP TERMINÉ! Tous les fichiers ont été générés avec succès.")
print("="*80)
