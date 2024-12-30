Formulaire de Diagnostic de la Tuberculose

Ce projet est une application web conçue pour collecter des informations sur les patients et télécharger des images afin d'aider au diagnostic de la tuberculose. Développée avec Flask et intégrée à un modèle d'apprentissage automatique, l'application permet une soumission efficace des données et une analyse rapide.

Fonctionnalités principales :

Collecte des informations des patients via un formulaire interactif.
Téléchargement d'images médicales pour analyse.
Validation des champs du formulaire pour garantir l'exactitude des données.
Intégration avec un modèle d'IA pour prédire la présence de la tuberculose.

Comment ça fonctionne :

Saisie des données : L'utilisateur remplit le formulaire en ligne avec les informations du patient (nom, date de naissance, antécédents, etc.).

Téléchargement d'images : Une ou plusieurs images médicales (comme des radiographies) peuvent être ajoutées via un champ dédié.

Validation des données : Le formulaire vérifie que tous les champs obligatoires sont remplis.

Envoi des données : Une fois soumises, les informations et les fichiers sont envoyés au serveur Flask.

Analyse par IA : Le serveur utilise un modèle de machine learning pour analyser les images et fournir des résultats diagnostiques.

Technologies utilisées :

Back-end : Flask (Python)
Front-end : HTML, CSS, JavaScript
Modèle IA : TensorFlow/Keras (ou tout autre framework d'apprentissage automatique)
