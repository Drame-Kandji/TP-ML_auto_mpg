import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Obtenir les données
#Commencez par télécharger et importer l'ensemble de données à l'aide de pandas :

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# Nettoyer les données
#L'ensemble de données contient quelques valeurs inconnues :
print(dataset.isna().sum())

# Supprimez ces lignes pour que ce didacticiel initial reste simple :
dataset = dataset.dropna()

#La colonne "Origin" est catégorique et non numérique. L'étape suivante consiste donc à encoder à chaud les valeurs de la colonne avec pd.get_dummies .
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())

# Diviser les données en ensembles d'apprentissage et de test
#Maintenant, divisez l'ensemble de données en un ensemble d'apprentissage et un ensemble de test. Vous utiliserez l'ensemble de test dans l'évaluation finale de vos modèles.
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset.shape)
print(test_dataset.shape)

#Inspectez les données
#Examinez la distribution conjointe de quelques paires de colonnes de l'ensemble d'apprentissage.
#La rangée du haut suggère que l'efficacité énergétique (MPG) est fonction de tous les autres paramètres. Les autres rangées indiquent qu'elles sont fonction l'une de l'autre.
pairplot = sns.pairplot(
    train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']],
    diag_kind='kde'
)

# Sauvegarde utile si l'environnement n'a pas d'affichage graphique.
pairplot.savefig('pairplot_train_dataset.png', dpi=150, bbox_inches='tight')
plt.show()

# Vérifions également les statistiques globales. Notez que chaque fonctionnalité couvre une plage très différente :
print(train_dataset.describe().transpose())


#Séparer les entités des étiquettes
#Séparez la valeur cible (le "libellé") des caractéristiques. Cette étiquette est la valeur que vous entraînerez le modèle à prédire.

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#Normalisation
#Dans le tableau des statistiques, il est facile de voir à quel point les plages de chaque fonctionnalité sont différentes :
print(train_dataset.describe().transpose()[['mean', 'std']])

#La couche de normalisation 
#Le tf.keras.layers.Normalization est un moyen propre et simple d'ajouter la normalisation des fonctionnalités dans votre modèle.
#La première étape consiste à créer le calque :
normalizer = tf.keras.layers.Normalization(axis=-1)

# Ensuite, adaptez l'état de la couche de prétraitement aux données en appelant Normalization.adapt :
train_features_array = np.array(train_features, dtype=np.float32)
normalizer.adapt(train_features_array)

# Calculez la moyenne et la variance, et stockez-les dans la couche :
print(normalizer.mean.numpy())

#Lorsque la couche est appelée, elle renvoie les données d'entrée, chaque entité étant normalisée indépendamment :
first = np.array(train_features[:1], dtype=np.float32)

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

# Le nombre d' entrées peut être défini soit par l'argument input_shape , soit automatiquement lorsque le modèle est exécuté pour la première fois.
#Tout d'abord, créez un tableau NumPy composé des fonctionnalités 'Horsepower' . Ensuite, instanciez tf.keras.layers.Normalization et adaptez son état aux données de horsepower :
horsepower = np.array(train_features['Horsepower'])

horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

# Construisez le modèle séquentiel Keras :
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.summary()

# Ce modèle prédira 'MPG' à partir de 'Horsepower' .
#Exécutez le modèle non formé sur les 10 premières valeurs de "puissance". La sortie ne sera pas bonne, mais notez qu'elle a la forme attendue de (10, 1) :

horsepower_model.predict(horsepower[:10])
print(horsepower_model.predict(horsepower[:10]))

#Une fois le modèle construit, configurez la procédure d'entraînement à l'aide de la méthode Keras Model.compile . Les arguments les plus importants à compiler sont le loss et l' optimizer , car ceux-ci définissent ce qui sera optimisé ( mean_absolute_error ) et comment (en utilisant le tf.keras.optimizers.Adam ).

horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

# Utilisez Keras Model.fit pour exécuter la formation pendant 100 époques :
history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# Visualisez la progression de l'entraînement du modèle à l'aide des statistiques stockées dans l'objet d' history :
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# Visualisez la progression de l'entraînement du modèle à l'aide des statistiques stockées dans l'objet d' history :
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.savefig('loss_plot.png', dpi=150, bbox_inches='tight')
  plt.show()


plot_loss(history)

# Collectez les résultats sur le jeu de test pour plus tard :
test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

#Puisqu'il s'agit d'une régression à variable unique, il est facile de visualiser les prédictions du modèle en fonction de l'entrée :
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()
  plt.savefig('horsepower_plot.png', dpi=150, bbox_inches='tight')
  plt.show()

plot_horsepower(x, y)

# Régression linéaire avec plusieurs entrées
#Vous pouvez utiliser une configuration presque identique pour effectuer des prédictions basées sur plusieurs entrées. Ce modèle fait toujours le même sauf que est une matrice et est un vecteur.
#Créez à nouveau un modèle séquentiel Keras en deux étapes avec la première couche étant le normalizer ( tf.keras.layers.Normalization(axis=-1) ) que vous avez défini précédemment et adapté à l'ensemble de données :
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

# Lorsque vous appelez Model.predict sur un lot d'entrées, il produit des sorties units=1 pour chaque exemple :
linear_model.predict(train_features[:10])

#Lorsque vous appelez le modèle, ses matrices de poids seront construites - vérifiez que les poids du kernel (le m  dans y = mx+b ) ont la forme (9, 1) :
linear_model.layers[1].kernel

# Configurez le modèle avec Keras Model.compile et entraînez-vous avec Model.fit pour 100 époques :
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    # Suppress logging.
    verbose=0,
    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)

# L'utilisation de toutes les entrées de ce modèle de régression permet d'obtenir une erreur de formation et de validation beaucoup plus faible que le modèle horsepower_model , qui avait une entrée :
plot_loss(history)

# Collectez les résultats sur le jeu de test pour plus tard :
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

#Les deux modèles utiliseront la même procédure de formation, de sorte que la méthode de compile est incluse dans la fonction build_and_compile_model ci-dessous.
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

# Régression utilisant un DNN et une seule entrée
#Créez un modèle DNN avec uniquement 'Horsepower' comme entrée et horsepower_normalizer (défini précédemment) comme couche de normalisation :
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

# Ce modèle a un peu plus de paramètres entraînables que les modèles linéaires :
dnn_horsepower_model.summary()

# Entraînez le modèle avec Keras Model.fit :
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

# Ce modèle fait légèrement mieux que le modèle linéaire à entrée unique horsepower_model :
plot_loss(history)

#Si vous tracez les prédictions en fonction de 'Horsepower' , vous devriez remarquer comment ce modèle tire parti de la non-linéarité fournie par les couches cachées :
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

# Collectez les résultats sur le jeu de test pour plus tard :
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)

# Régression utilisant un DNN et plusieurs entrées
#Répétez le processus précédent en utilisant toutes les entrées. Les performances du modèle s'améliorent légèrement sur l'ensemble de données de validation.
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()


history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

# Recueillez les résultats sur l'ensemble de test :
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# Performance
#Étant donné que tous les modèles ont été entraînés, vous pouvez examiner les performances de leurs ensembles de test :
pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T

# Faire des prédictions
#Vous pouvez maintenant faire des prédictions avec le dnn_model sur l'ensemble de test à l'aide de Keras Model.predict et examiner la perte :
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

# Il semble que le modèle prédit raisonnablement bien.
#Maintenant, vérifiez la distribution des erreurs :*
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')

#Si vous êtes satisfait du modèle, enregistrez-le pour une utilisation ultérieure avec Model.save :
dnn_model.save('dnn_model.keras')

# Si vous rechargez le modèle, il donne une sortie identique :
reloaded = tf.keras.models.load_model('dnn_model.keras')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T

# Conclusion
#Ce cahier a introduit quelques techniques pour gérer un problème de régression. Voici quelques conseils supplémentaires qui peuvent vous aider :
#L'erreur quadratique moyenne (MSE) ( tf.losses.MeanSquaredError ) et l'erreur absolue moyenne (MAE) ( tf.losses.MeanAbsoluteError ) sont des fonctions de perte courantes utilisées pour les problèmes de régression. MAE est moins sensible aux valeurs aberrantes. Différentes fonctions de perte sont utilisées pour les problèmes de classification.
#De même, les mesures d'évaluation utilisées pour la régression diffèrent de la classification.
#Lorsque les entités de données d'entrée numériques ont des valeurs avec des plages différentes, chaque entité doit être mise à l'échelle indépendamment dans la même plage.
#Le surajustement est un problème courant pour les modèles DNN, même si ce n'était pas un problème pour ce didacticiel. Consultez le didacticiel de surajustement et de sous-ajustement pour obtenir de l'aide à ce sujet.













