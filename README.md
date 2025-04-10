# Titanic
Prédiction sur les survivants du titanic
# 🚢 Prédiction de Survie - Titanic (Kaggle)

Ce projet utilise un modèle de réseau de neurones pour prédire la survie des passagers du Titanic à partir du célèbre dataset de Kaggle.

Le notebook comprend :

- Une **exploration des données (EDA)**,
- Le **prétraitement complet** (gestion des valeurs manquantes, encodage, normalisation),
- L'entraînement d'un **modèle de classification binaire** avec TensorFlow/Keras,
- La génération d’un **fichier de soumission** pour Kaggle.

---

## 📊 Données utilisées

Les données proviennent du challenge Kaggle [Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). Les fichiers utilisés sont :

- `train.csv` : données d'entraînement,
- `test.csv` : données de test pour soumission,
- `submission.csv` : prédictions générées par le modèle.

---

## 🔧 Fonctionnalités

- Visualisation : histogrammes, corrélations, heatmaps de valeurs manquantes,
- Nettoyage des données et création de nouvelles variables (`Has_Cabin`),
- Encodage de variables catégorielles (`Sex`, `Embarked`),
- Normalisation (`Age`, `Fare`),
- Modélisation avec **réseau de neurones profond** (2 couches cachées, dropout),
- Évaluation via **matrice de confusion**, **rapport de classification**, **accuracy**.

---

## ⚙️ Utilisation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/ton-utilisateur/Titanic_Prediction.git
   cd Titanic_Prediction
   
##Assurez vous d’avoir les fichiers train.csv et test.csv dans le même dossier que le script.
