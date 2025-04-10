import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Charger les données
data = pd.read_csv('/home/mouhamed-bachir.cisse/ResauxNeurones/train.csv')
print(type(data))
print(data.head())

# Informations générales sur les données
#print(data.info())

# Statistiques descriptives
#print(data.describe())
# Répartition des passagers par classe
print(data['Pclass'].value_counts())

# Répartition des passagers par sexe
print(data['Sex'].value_counts())

# Répartition des passagers par port d'embarquement
print(data['Embarked'].value_counts())

# Répartition des survivants et non-survivants
print(data['Survived'].value_counts())

# Histogramme pour la distribution de l'âge
sns.histplot(data['Age'].dropna(), kde=True, bins=30)
plt.title("Distribution de l'âge des passagers")
plt.xlabel("Âge")
plt.ylabel("Nombre de passagers")
plt.show()

# Visualisation de la répartition des survivants et non-survivants
sns.countplot(data=data, x='Survived')
plt.title("Répartition des survivants et non-survivants")
plt.xlabel("Survécu (1 = Oui, 0 = Non)")
plt.ylabel("Nombre de passagers")
plt.show()

# Répartition par sexe et survie
sns.countplot(data=data, x='Sex', hue='Survived')
plt.title("Survie par sexe")
plt.xlabel("Sexe")
plt.ylabel("Nombre de passagers")
plt.legend(["Non-Survécu", "Survécu"])
plt.show()

# Répartition par classe et survie
sns.countplot(data=data, x='Pclass', hue='Survived')
plt.title("Survie par classe de passagers")
plt.xlabel("Classe")
plt.ylabel("Nombre de passagers")
plt.legend(["Non-Survécu", "Survécu"])
plt.show()

#  uniquement les colonnes numériques pour  la matrice de corrélation
numeric_data = data.select_dtypes(include=[np.number])

#  la matrice de corrélation
correlation_matrice = numeric_data.corr()

# Affichage des corrélations avec la colonne 'Survived'
print(correlation_matrice["Survived"].sort_values(ascending=False))
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrice, cmap="coolwarm", annot=True)
plt.title("Matrice de corrélation des survivants")
plt.show()

# Affichage du  pourcentage des survivants
survived_count = data['Survived'].value_counts(normalize=True) * 100
print(survived_count)
# Comptage des valeurs manquantes par colonne
print("Les valeurs manquantes par colonne sont")
missing_values = data.isnull().sum()
print(missing_values)

# Visualiser les valeurs manquantes
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Valeurs manquantes dans les données")
plt.show()

print("valeurs manquantes de Age avant remplacement")
print(data['Age'].isnull().sum())
# Remplacement des valeurs manquantes de la colonne 'Age' par la médiane
data['Age'].fillna(data['Age'].median(), inplace=True)
print("valeurs manquantes de Age après remplacement")
# Vérification de  si les valeurs manquantes de 'Age' ont été remplacées
print(data['Age'].isnull().sum())

# Créon d'une nouvelle colonne 'Has_Cabin' qui vaut 1 si une cabine est attribuée, sinon 0
data['Has_Cabin'] = data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

print("Nouvelle colonne Has_Cabin")
print(data['Has_Cabin'].value_counts())
# Supprimer la colonne 'Cabin' car elle contient trop de valeurs manquantes
data.drop('Cabin', axis=1, inplace=True)

print("valeurs manquantes de Embarked avant remplacement")
print(data['Embarked'].isnull().sum())
# Remplacer les valeurs manquantes de 'Embarked' par la valeur la plus fréquente (mode)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Vérifier si les valeurs manquantes de 'Embarked' ont été remplacées
print("valeurs manquantes de Embarked après remplacement")
print(data['Embarked'].isnull().sum())
# Vérifier s'il reste des valeurs manquantes
print('Après rectification des valeurs manquantes, nous avons')
print(data.isnull().sum())
# Remplacer les valeurs manquantes de 'Fare' par la médiane
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Transformer 'Sex' en valeurs numériques (0 pour 'male', 1 pour 'female')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
print(data['Sex'].value_counts())
# Transformer 'Embarked' en valeurs numériques
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
print(data.value_counts())
# Paramètres des gaussiennes
mean_class_0 = [0, 0]  # Centre de la classe 0
mean_class_1 = [0.5, 0]  # Centre de la classe 1
cov_matrix = [[0.05, 0], [0, 0.05]]  # Matrice de covariance identique pour les deux classes

# Générer des données pour chaque classe
num_samples_per_class = 60
class_0_data = np.random.multivariate_normal(mean_class_0, cov_matrix, num_samples_per_class)
class_1_data = np.random.multivariate_normal(mean_class_1, cov_matrix, num_samples_per_class)

# Créer des étiquettes pour chaque classe (0 pour la classe 0 et 1 pour la classe 1)
class_0_labels = np.zeros(num_samples_per_class, dtype=int)
class_1_labels = np.ones(num_samples_per_class, dtype=int)
# Fusionner les données et les étiquettes pour former l'ensemble de données complet
X = np.vstack((class_0_data, class_1_data))
y = np.hstack((class_0_labels, class_1_labels))

# Afficher les données
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Caractéristique 1')
plt.ylabel('Caractéristique 2')
plt.title('Échantillon de données avec deux classes')
# Définir les axes de même taille
plt.axis('equal')

plt.show()

# Sélectionner les caractéristiques (features) et les étiquettes (labels)
Z = data.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], axis=1)
w = data['Survived']
# Normalisation des caractéristiques numériques (Age, Fare)
scaler = StandardScaler()
Z[['Age', 'Fare']] = scaler.fit_transform(Z[['Age', 'Fare']])

# Division des données en ensemble d'entraînement et de test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(Z, w, test_size=0.2, random_state=42)

# Afficher la forme des ensembles d'entraînement et de test après leur définition
print("Forme de l'ensemble d'entraînement :", X_train.shape)
print("Forme de l'ensemble de test :", X_test.shape)
# Créer le modèle
model = Sequential()

# Ajouter la première couche cachée avec 16 neurones, et la fonction d'activation ReLU
model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu'))
from tensorflow.keras.layers import Dropout
# Ajouter une deuxième couche cachée
model.add(Dense(16, activation='relu', input_dim=2))
model.add(Dropout(0.3))

# Ajouter la couche de sortie avec 1 neurone (sortie binaire : survie ou non)
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Afficher la structure du modèle
model.summary()

# Entraîner le modèle avec les données d'entraînement
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Afficher l'historique d'entraînement
plt.plot(history.history['accuracy'], label='Précision entraînement')
plt.plot(history.history['val_accuracy'], label='Précision validation')
plt.xlabel('Épochs')
plt.ylabel('Précision')
plt.legend()
plt.show()

# Évaluation du modèle sur l'ensemble de test
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int32")

print(" Matrice de confusion et le rapport de classification")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# La précision globale
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")
# Les données de test de Kaggle (test.csv)
test_data = pd.read_csv('/home/mouhamed-bachir.cisse/ResauxNeurones/test.csv')

# Préparation des données de test comme nous l'avons fait pour les données d'entraînement
test_data['Age'].fillna(data['Age'].median(), inplace=True)
test_data['Fare'].fillna(data['Fare'].median(), inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data = pd.get_dummies(test_data, columns=['Embarked'], drop_first=True)

# Sélectionner les caractéristiques et normaliser
X_kaggle = test_data.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
X_kaggle[['Age', 'Fare']] = scaler.transform(X_kaggle[['Age', 'Fare']])
X_kaggle = X_kaggle.reindex(columns=X_train.columns, fill_value=0)

# Prédictions sur l'ensemble test
kaggle_predictions = model.predict(X_kaggle)
kaggle_predictions = (kaggle_predictions > 0.5).astype(int).flatten()
# Créer la matrice de confusion
cm = confusion_matrix(y_test, y_pred )

# Afficher la matrice de confusion
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Survécu', 'Survécu'], yticklabels=['Non-Survécu', 'Survécu'])
plt.xlabel('Prédictions')
plt.ylabel('Vérités réelles')
plt.title('Matrice de Confusion')
plt.show()
# Fichier de soumission
submission = pd.DataFrame({
    "PassengerId": test_data['PassengerId'],
    "Survived": kaggle_predictions
})
submission.to_csv('submission.csv', index=False)

print("Fichier de soumission créé avec succès.")
