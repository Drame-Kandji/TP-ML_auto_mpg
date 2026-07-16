# Auto MPG Predictor

Application web de **prédiction de consommation de carburant (MPG)** basée sur un réseau de neurones profond (DNN) entraîné avec TensorFlow, intégré dans une interface Django.

> **MPG** = *Miles Per Gallon* plus la valeur est élevée, plus le véhicule est économe en carburant.

---

## Aperçu

![Aperçu de l'application](https://img.shields.io/badge/statut-opérationnel-brightgreen)

L'utilisateur remplit un formulaire avec les caractéristiques du véhicule (cylindres, cylindrée, puissance, poids, accélération, année, origine) et obtient instantanément une estimation MPG générée par un modèle TensorFlow pré-entraîné.


## Architecture du projet

```
AutoMPG-project/
├── auto_mpg_project/          # Configuration Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── models/                    # Modèle entraîné & scripts
│   ├── dnn_model.keras        # Modèle sauvegardé (chargé par Django)
│   ├── regression.py          # Script d'entraînement du modèle
│   ├── loss_plot.png          # Courbe d'apprentissage
│   ├── horsepower_plot.png    # Régression puissance/MPG
│   └── pairplot_train_dataset.png
├── predictor/                 # Application Django
│   ├── forms.py               # Formulaire de prédiction
│   ├── model_service.py       # Service de chargement & inférence
│   ├── views.py               # Vue unique de prédiction
│   └── templates/predictor/
│       └── index.html         # Page d'accueil
├── static/predictor/
│   └── styles.css             # Styles de l'interface
├── db.sqlite3                 # Base de données (non utilisée)
├── manage.py                  # Point d'entrée Django
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

---

## Guide de lancement

### Prérequis

- **Python 3.10+**
- **pip** (gestionnaire de paquets Python)
- **Git** (optionnel, pour cloner le dépôt)

### 1. Cloner le projet

```bash
git clone <url-du-depot>
cd AutoMPG-project
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python3 -m venv venv
source venv/bin/activate    # Linux / macOS
# ou
venv\Scripts\activate       # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

> **Note TensorFlow** : L'installation de TensorFlow peut prendre plusieurs minutes. Sur certains systèmes, il peut être nécessaire d'installer des dépendances système supplémentaires (CUDA, cuDNN) pour l'accélération GPU. Consultez la [documentation officielle](https://www.tensorflow.org/install) si nécessaire.

### 4. Lancer le serveur de développement

```bash
python manage.py runserver
```

### 5. Ouvrir l'application

Rendez-vous sur [http://127.0.0.1:8000](http://127.0.0.1:8000) dans votre navigateur.


## Utilisation

1. Remplissez les champs du formulaire avec les caractéristiques du véhicule :
   - **Cylindres** : nombre de cylindres (ex: 4, 6, 8)
   - **Cylindrée** : volume du moteur en pouces cubes (ex: 150.0)
   - **Puissance** : puissance en chevaux (ex: 100.0)
   - **Poids** : poids en livres (ex: 3000.0)
   - **Accélération** : temps de 0 à 60 mph en secondes (ex: 15.0)
   - **Année du modèle** : sélectionnez une date dans l'année souhaitée (seule l'année est utilisée)
   - **Origine** : USA, Europe ou Japan

2. Cliquez sur **"Lancer la prédiction"**

3. Le résultat s'affiche instantanément dans la colonne de gauche avec la valeur estimée en MPG.


## Ré-entraîner le modèle

Si vous souhaitez ré-entraîner le modèle vous-même :

```bash
cd models
python regression.py
```

Cela générera un nouveau fichier `dnn_model.keras` et produira les graphiques d'analyse (`loss_plot.png`, `horsepower_plot.png`, `pairplot_train_dataset.png`).

> Assurez-vous que l'environnement virtuel est activé et que TensorFlow est installé.
