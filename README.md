# Analyse vidéo

Ce dépôt rassemble les notebooks d’acquisition et de segmentation, puis d’estimation du flux optique sur une séquence vidéo.


### TP1 — Acquisition, segmentation spatiale et temporelle

Notebook : [`TP1/TP1_VideoAnalysis.ipynb`](TP1/TP1_VideoAnalysis.ipynb)

- Chargement d’une vidéo avec OpenCV, extraction et sauvegarde des images
- Segmentation spatiale : seuillage heuristique, méthode d’Otsu, mélange de gaussiennes
- Segmentation temporelle : différence d’images, différence sur trois images, soustraction de fond adaptative


### Estimation du flux optique

Trois modèles pré-entraînés, **en inférence uniquement** (sans réentraînement), sur la vidéo `video voiture 7s.mp4` :

| Notebook | Modèle | Environnement |
|---|---|---|
| [`01_RAFT.ipynb`](Estimation%20du%20flux%20optique/01_RAFT.ipynb) | [RAFT](https://github.com/princeton-vl/RAFT) (ECCV 2020) | Google Colab (GPU) |
| [`02_MemFlow.ipynb`](Estimation%20du%20flux%20optique/02_MemFlow.ipynb) | [MemFlow](https://github.com/DQiaole/MemFlow) (CVPR 2024) | Google Colab (GPU) |
| [`03_Freeflow.ipynb`](Estimation%20du%20flux%20optique/03_Freeflow.ipynb) | [FreeFlow](https://github.com/msu-video-group/freeflow) | Kaggle / GPU |

Chaque notebook mesure le temps d’exécution, la mémoire GPU, visualise le flux (couleurs, magnitude, champ de vecteurs) et évalue la précision sur le benchmark **Middlebury** (EPE, Fl-all).

## Exécution

Les notebooks de flux optique sont prévus pour un runtime **GPU** (T4 ou équivalent).

1. Ouvrir le notebook dans Colab ou Kaggle.
2. Monter le stockage (Google Drive ou dataset) et indiquer le chemin de la vidéo.
3. Exécuter toutes les cellules dans l’ordre.

Le TP1 s’exécute de la même manière, avec OpenCV, NumPy, Matplotlib et scikit-learn.

## Contributeurs

| AYEB Farah | BEN AMOR Hanine | MELKI Meriem | 
