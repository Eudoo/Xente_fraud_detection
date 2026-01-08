# Script PowerShell pour planifier le retraining automatique sur Windows
# Utilise le Planificateur de tâches Windows

$ScriptPath = Join-Path $PSScriptRoot "automated_retraining.py"
$PythonPath = (Get-Command python).Source
$LogPath = Join-Path $PSScriptRoot "logs\scheduled_retraining.log"

Write-Host "========================================" -ForegroundColor Blue
Write-Host "Configuration du Retraining Automatique" -ForegroundColor Blue
Write-Host "========================================" -ForegroundColor Blue

# Vérifier que le script existe
if (-not (Test-Path $ScriptPath)) {
    Write-Host "Erreur: Le script automated_retraining.py n'existe pas" -ForegroundColor Red
    exit 1
}

Write-Host "`nScript trouvé: $ScriptPath" -ForegroundColor Green
Write-Host "Python: $PythonPath" -ForegroundColor Green

# Créer le dossier logs s'il n'existe pas
$LogDir = Join-Path $PSScriptRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Configuration de la tâche planifiée
$TaskName = "FraudDetection_AutoRetraining"
$TaskDescription = "Retraining automatique du modèle de détection de fraude"

# Créer l'action (exécuter le script Python)
$Action = New-ScheduledTaskAction -Execute $PythonPath `
    -Argument "$ScriptPath" `
    -WorkingDirectory $PSScriptRoot

# Créer le déclencheur (tous les jours à 2h00)
$Trigger = New-ScheduledTaskTrigger -Daily -At 2am

# Créer les paramètres
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Enregistrer la tâche
try {
    # Supprimer la tâche si elle existe déjà
    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($ExistingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "`nTâche existante supprimée" -ForegroundColor Yellow
    }
    
    # Créer la nouvelle tâche
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Description $TaskDescription `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -User $env:USERNAME `
        -RunLevel Highest | Out-Null
    
    Write-Host "`n✓ Tâche planifiée créée avec succès!" -ForegroundColor Green
    Write-Host "`nDétails de la tâche:" -ForegroundColor Blue
    Write-Host "  Nom: $TaskName"
    Write-Host "  Planification: Tous les jours à 2h00"
    Write-Host "  Script: $ScriptPath"
    Write-Host "`nPour gérer la tâche:"
    Write-Host "  - Ouvrir le Planificateur de tâches Windows"
    Write-Host "  - Rechercher: $TaskName"
    Write-Host "`nPour voir les logs:"
    Write-Host "  Get-Content logs\retraining_*.log -Tail 50"
    
} catch {
    Write-Host "`nErreur lors de la création de la tâche planifiée:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Blue
Write-Host "Configuration terminée!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Blue
