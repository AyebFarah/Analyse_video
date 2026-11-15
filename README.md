# Estimation du flux optique : du raffinement itératif à la mémoire

**RAFT · MemFlow · FreeFlow** : étude de trois architectures de l'état de l'art et test sur notre propre vidéo.

Projet d'analyse vidéo réalisé par **Hanine BEN AMOR**, **Farah AYEB** et **Meriem MELKI**.

<p align="center">
  <img src="documentation/slides/00_titre.png" width="720" alt="Page de titre de la présentation">
</p>

On étudie trois modèles qui estiment le flux optique, c'est-à-dire le déplacement de chaque pixel entre deux images d'une vidéo :

| Modèle | Publication | Idée principale |
|---|---|---|
| **RAFT** | Teed & Deng, ECCV 2020 | un volume de corrélation entre tous les pixels, puis 12 à 32 itérations de correction |
| **MemFlow** | Dong & Fu, CVPR 2024 | RAFT, plus une **mémoire** du mouvement des images précédentes |
| **FreeFlow** | Bargatin et al., ECCV 2026 | un transformer simple, **en une seule passe**, sans aucun module spécial pour le flux |

Pour chaque modèle, on explique l'architecture, l'entraînement et les résultats publiés, puis on le teste **en inférence seule** (poids publiés par les auteurs, sans réentraînement) sur la même vidéo : une petite voiture filmée de dessus, 576 × 1024 pixels, 30 images par seconde, environ 7 secondes.

---

## Sommaire

1. [Le flux optique et ses métriques](#1-le-flux-optique-et-ses-métriques)
2. [RAFT](#2-raft)
3. [MemFlow](#3-memflow)
4. [FreeFlow](#4-freeflow)
5. [Comparaison des trois modèles](#5-comparaison-des-trois-modèles)
6. [Reproduire nos résultats](#6-reproduire-nos-résultats)
7. [Contenu du dépôt](#7-contenu-du-dépôt)
8. [Références](#8-références)

---

## 1. Le flux optique et ses métriques

Le flux optique associe à chaque pixel un vecteur de déplacement **(u, v)** : u est le déplacement horizontal, v le vertical, en pixels.
- **Entrée** : deux images I(t) et I(t+1), de taille H × W × 3.
- **Sortie** : un champ dense de taille H × W × 2.
- **Codage couleur** (Middlebury) : la teinte donne la direction, l'intensité la vitesse, et le blanc veut dire « immobile ».

<p align="center">
  <img src="documentation/slides/01_flux_optique.png" width="49%" alt="Le flux optique">
  <img src="documentation/slides/02_metriques.png" width="49%" alt="Les métriques">
</p>

| Métrique | Définition | Sens |
|---|---|---|
| **EPE** (End-Point Error) | distance moyenne, en pixels, entre la flèche prédite et la vraie flèche | plus bas = mieux |
| **Fl-all** | % de pixels faux : erreur > 3 px **et** > 5 % du vrai déplacement | plus bas = mieux |
| **1px** | % de pixels faux de plus de 1 pixel (Spring) | plus bas = mieux |
| **WAUC** | % de pixels justes, pour toutes les tolérances de 0 à 5 px (Spring) | plus haut = mieux |
| **Erreur photométrique** | on déplace l'image 2 avec le flux et on la compare à l'image 1 : moyenne de \|I₁ − I₂(x+u, y+v)\| | plus bas = mieux |

Notre vidéo n'a pas de vérité terrain (on ne connaît pas le vrai mouvement de chaque pixel). On mesure donc sa qualité avec l'**erreur photométrique**. Pour avoir une vraie mesure de précision, on teste aussi chaque modèle sur **Middlebury** (8 séquences avec vrai flux connu, paire frame10 → frame11), avec le même protocole pour les trois modèles.

---

## 2. RAFT

> *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, Z. Teed et J. Deng, ECCV 2020 ([arXiv:2003.12039](https://arxiv.org/abs/2003.12039)).

RAFT a trois blocs :
1. **Les encodeurs** décrivent chaque bloc de 8 × 8 pixels par 256 nombres, et un 2ᵉ encodeur décrit le contexte de l'image 1.
2. **Le volume de corrélation** compare chaque pixel de l'image 1 à tous les pixels de l'image 2.
3. **La mise à jour itérative** part d'un flux nul et le corrige 12 à 32 fois, avec le même petit réseau récurrent (ConvGRU).

<p align="center">
  <img src="documentation/slides/10_raft_architecture.png" width="720" alt="Architecture de RAFT">
</p>

<details>
<summary><b>Voir le détail des blocs (corrélation, ConvGRU)</b></summary>
<p align="center">
  <img src="documentation/slides/11_raft_correlation.png" width="49%" alt="Volume de corrélation">
  <img src="documentation/slides/12_raft_gru.png" width="49%" alt="Mise à jour récurrente">
</p>
</details>

### Les formules

**Le volume de corrélation** : g(I₁) et g(I₂) sont les caractéristiques des deux images (résolution 1/8, D = 256). On compare chaque pixel (i, j) de l'image 1 à chaque pixel (k, l) de l'image 2 par un produit scalaire :

$$
C_{ijkl} = \frac{1}{\sqrt{D}} \sum_{h=1}^{D} g(I_1)_{ijh}\, g(I_2)_{klh}, \qquad C \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times \frac{H}{8} \times \frac{W}{8}}
$$

On construit ensuite une **pyramide à 4 niveaux** en moyennant les deux dernières dimensions (noyaux 1, 2, 4, 8), pour voir aussi les grands déplacements. À chaque itération, RAFT ne lit qu'une fenêtre de 9 × 9 (rayon r = 4) autour de la position estimée, dans chaque niveau.

**La mise à jour (ConvGRU)** : xₜ regroupe les corrélations lues, le flux actuel et le contexte de l'image 1.

$$
z_t = \sigma\big(\mathrm{Conv}_{3\times3}([h_{t-1}, x_t], W_z)\big), \qquad r_t = \sigma\big(\mathrm{Conv}_{3\times3}([h_{t-1}, x_t], W_r)\big)
$$

$$
\tilde h_t = \tanh\big(\mathrm{Conv}_{3\times3}([r_t \odot h_{t-1}, x_t], W_h)\big), \qquad h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t
$$

Deux convolutions prédisent la correction Δf à partir de hₜ, puis f₍ₖ₊₁₎ = fₖ + Δf. Le flux à 1/8 est enfin remis à pleine résolution par un **sur-échantillonnage convexe** (×8, chaque pixel est une combinaison de ses 9 voisins).

**La perte** : on compare au vrai flux le flux de **chaque** itération, avec plus de poids sur les dernières (γ = 0,8, N itérations) :

$$
\mathcal{L} = \sum_{i=1}^{N} \gamma^{N-i} \, \lVert f_{gt} - f_i \rVert_1
$$

### Entraînement et données

1. **FlyingChairs** (images synthétiques de chaises) : 100 k itérations, batch de 12.
2. **FlyingThings3D** (objets 3D synthétiques) : 100 k itérations, batch de 6. Ce modèle s'appelle « C+T » (checkpoint `raft-things`).
3. **Fine-tuning Sintel** : 100 k itérations sur un mélange Sintel + KITTI-2015 + HD1K. C'est le checkpoint `raft-sintel` utilisé sur notre vidéo.
4. **Fine-tuning KITTI** : 50 k itérations de plus, à partir du modèle Sintel (checkpoint `raft-kitti`).

Optimiseur AdamW, 2 GPU 2080 Ti. En évaluation, les auteurs utilisent 32 itérations sur Sintel et 24 sur KITTI.

### Résultats publiés

| Entraînement | Évaluation | RAFT (2 images) | RAFT (warm-start) |
|---|---|---|---|
| C+T (généralisation) | Sintel train, Clean / Final (EPE) | 1,43 / 2,71 | – |
| C+T (généralisation) | KITTI-15 train, EPE / Fl-all | 5,04 / 17,4 % | – |
| C+T+S+K+H (fine-tuning) | Sintel test, Clean / Final (EPE) | 1,94 / 3,18 | **1,61 / 2,86** |
| C+T+S+K+H (fine-tuning) | KITTI-15 test, Fl-all | **5,10 %** | – |

- En 2020, RAFT est 1er sur Sintel (−30 % d'erreur sur Final par rapport au meilleur résultat publié) et 1er sur KITTI.
- Le **warm-start** (initialiser avec le flux projeté de la paire précédente) n'est possible que parce que RAFT travaille à une seule résolution. Il gagne environ 0,3 px sur Sintel.
- Les ablations du papier montrent l'importance du ConvGRU (EPE 1,63 contre 2,04 avec de simples convolutions), du partage des poids entre itérations et de la corrélation sur **toutes** les paires de pixels (1,63 contre 2,91 si on la limite à ±32 px).

### RAFT sur notre vidéo

| Mesure (GPU Tesla T4) | Valeur |
|---|---|
| Modèle / itérations | `raft-sintel` / 20 |
| Temps par paire | 705 ms (1,4 paire/s) |
| Pic mémoire GPU | 726 Mo, dont 324 Mo pour la corrélation |
| Paramètres | 5,26 M |
| Erreur photométrique | −59 % avec le flux |
| Middlebury | EPE 0,28 px · Fl-all 0,94 % |

<p align="center">
  <img src="documentation/figures/RAFT/RAFT_frames_flow_heatmaps.png" width="640" alt="Flux de RAFT sur notre vidéo">
</p>

- La voiture est segmentée proprement : un seul mouvement vers le haut (violet), et le fond reste immobile (blanc).
- Avec **12 itérations au lieu de 20**, la qualité est la même, et le temps passe de 713 à 500 ms (−30 %).
- **RAFT-small** (1 M paramètres) est 4 fois plus rapide (164 ms), mais un peu moins précis.
- 45 % de la mémoire sert à la matrice de corrélation : doubler la résolution multiplie ce coût par environ 16.
- Le **warm-start** n'apporte rien sur notre vidéo (erreur photométrique 0,407 en 2 images, 0,415 avec warm-start) ni sur Middlebury (EPE 0,282 contre 0,285) : le mouvement est faible et régulier, l'initialisation à zéro suffit.

<p align="center">
  <img src="documentation/figures/RAFT/RAFT_correlation_volume.png" width="49%" alt="Volume de corrélation de RAFT pour un pixel">
  <img src="documentation/figures/RAFT/RAFT_iterations_convergence.png" width="49%" alt="Convergence des itérations de RAFT">
  <br><sub>À gauche : la corrélation d'un pixel de la voiture avec toute l'image suivante, et la pyramide. À droite : la correction devient négligeable après quelques itérations.</sub>
</p>

<p align="center">
  <img src="documentation/slides/14_raft_compromis.png" width="49%" alt="Compromis vitesse / qualité">
  <img src="documentation/slides/15_raft_quand.png" width="49%" alt="Quand utiliser RAFT">
</p>

---

## 3. MemFlow

> *MemFlow: Optical Flow Estimation and Prediction with Memory*, Q. Dong et Y. Fu, CVPR 2024 ([arXiv:2404.04808](https://arxiv.org/abs/2404.04808)).

MemFlow reprend RAFT et ajoute une **mémoire** : une région qui ressemble à une région déjà vue peut réutiliser le mouvement trouvé avant.
1. **Les extracteurs** décrivent les images t et t+1, calculent les corrélations et construisent une représentation du mouvement et du contexte.
2. **La mémoire** garde le contexte et le mouvement de l'image précédente, et retrouve par attention les régions similaires.
3. **La mise à jour** raffine le flux en 15 itérations, en combinant le mouvement actuel et la mémoire, puis met à jour la mémoire.

<p align="center">
  <img src="documentation/slides/20_memflow_architecture.png" width="720" alt="Architecture de MemFlow">
</p>

**La formule de la mémoire** : chaque pixel actuel (question q) cherche les régions qui lui ressemblent, dans l'image actuelle et dans le passé (clés k), et récupère leurs mouvements (valeurs v) :

$$
q = f_c W_q, \quad k = [\,f_c W_k \,;\, k_m\,], \quad v = [\,f_m W_v \,;\, v_m\,]
$$

$$
f_{am} = f_m + \alpha \cdot \mathrm{Softmax}\!\left(\frac{\log_{n_{avg}}(L + HW)}{\sqrt{D_k}}\; q\,k^{\top}\right) v
$$

<p align="center">
  <img src="documentation/slides/21_memflow_memoire.png" width="49%" alt="La mémoire de MemFlow">
  <img src="documentation/slides/22_memflow_memoire_attention.png" width="49%" alt="Attention de la mémoire">
</p>

**Les détails de la formule et des blocs :**
- Les encodeurs (caractéristiques et contexte) ont la même structure que dans RAFT : 6 blocs résiduels, sortie à 1/8 de la résolution, et la même corrélation 4D que RAFT.
- **L'encodeur de mouvement** (6 blocs SKBlock, de grands noyaux de convolution) transforme le flux actuel et les corrélations lues en une caractéristique de mouvement f_m.
- **α** est un poids appris, initialisé à 0 : au début de l'entraînement, le modèle se comporte comme sans mémoire, puis il apprend combien utiliser le passé.
- Le terme log₍n_avg₎(L + HW) adapte l'attention à la **résolution** : avec plus de pixels, le softmax reste aussi « net » qu'à la résolution d'entraînement.
- **La mise à jour** est un GRU avec 2 SKBlocks : il reçoit le contexte, le mouvement f_m et le mouvement enrichi par la mémoire f_am, et prédit la correction du flux.
- **La taille de la mémoire** : par défaut, on ne garde que l'image précédente (l_max = 1). Les auteurs ont testé des mémoires plus longues, sans gain.
- **MemFlow-T** remplace les encodeurs par un transformer pré-entraîné sur ImageNet (Twins-SVT). Nous utilisons la version de base, `MemFlowNet_sintel`.

### Entraînement et données

1. **Pré-entraînement à 2 images** : FlyingChairs (120 k itérations, batch de 8), puis FlyingThings3D (150 k itérations, batch de 6).
2. **Entraînement vidéo à 3 images** (batch de 8) : 600 k itérations sur FlyingThings3D. C'est ici que la mémoire apprend à servir.
3. **Fine-tuning Sintel** : 600 k itérations sur un mélange Sintel + KITTI + HD1K + FlyingThings3D (checkpoint `MemFlowNet_sintel`, utilisé sur notre vidéo).
4. **Fine-tuning KITTI** (40 k itérations) et **Spring** (400 k itérations) pour les versions dédiées à ces benchmarks.

Optimiseur AdamW, 2 GPU A100. La perte est celle de RAFT (somme sur 12 itérations, avec un poids γ = 0,85 sur les dernières). En inférence, 15 itérations par défaut.

### Résultats publiés

| Entraînement | Évaluation | RAFT | MemFlow | MemFlow-T |
|---|---|---|---|---|
| C+T (généralisation) | Sintel train, Clean / Final (EPE) | 1,43 / 2,71 | 0,93 / 2,08 | **0,85 / 2,06** |
| C+T (généralisation) | KITTI-15 train, EPE / Fl-all | 5,04 / 17,4 % | 3,88 / 13,7 % | **3,38 / 12,8 %** |
| Fine-tuning | Sintel test, Clean / Final (EPE) | 1,61 / 2,86 * | **1,05** / 1,91 | 1,08 / **1,84** |
| Fine-tuning | KITTI-15 test, Fl-all | 5,10 % | 4,10 % | **3,88 %** |
| + Spring | Spring test, EPE / Fl | – | **0,471 / 1,42** | – |

\* RAFT avec warm-start.

- Par rapport à RAFT, MemFlow réduit l'erreur d'environ **35 %** sur Sintel test (Clean) et de 20 % sur KITTI.
- À sa sortie, MemFlow est **1er sur Spring** (1080p). VideoFlow, une méthode à 5 images, n'y tourne même pas : il dépasse les 80 Go de mémoire d'une A100.
- Vitesse annoncée : 5,6 images/s sur A100 en 1024 × 436, et 14,5 images/s avec seulement 5 itérations.

### MemFlow sur notre vidéo

| Mesure (GPU Tesla T4) | Valeur |
|---|---|
| Modèle / itérations | `MemFlowNet_sintel` / 15 |
| Temps par paire | 1 478 ms (0,68 paire/s) |
| Pic mémoire GPU | 3 794 Mo |
| Paramètres | 6,27 M |
| Erreur photométrique | 0,475 avec le flux, 1,213 sans (−61 %) |
| Middlebury | EPE 0,214 px · Fl-all 0,518 % |

<p align="center">
  <img src="documentation/figures/MemFlow/MemFlow_frames_flow_heatmaps.png" width="640" alt="MemFlow sur trois instants de la vidéo">
</p>

- Sur notre vidéo, **5 itérations suffisent** : la correction passe sous 0,01 px en 6 itérations, et l'erreur photométrique est la même de 2 à 24 itérations. Chaque itération coûte environ 92 ms.
- Sur Middlebury, le gain de la mémoire est négligeable (EPE 0,214 avec, 0,217 sans) : il n'y a que deux images par séquence, donc presque pas d'historique.

<p align="center">
  <img src="documentation/slides/24_memflow_iterations.png" width="49%" alt="Nombre d'itérations">
  <img src="documentation/slides/26_memflow_quand.png" width="49%" alt="Quand utiliser MemFlow">
</p>

---

## 4. FreeFlow

> *FreeFlow: A Bias-free Hierarchical Transformer for Optical Flow Estimation*, V. Bargatin, A. Yakovenko, K. Abud et D. Vatolin, ECCV 2026 ([arXiv:2609.11486](https://arxiv.org/abs/2609.11486)).

### Motivation : les modules faits à la main sont-ils nécessaires ?

RAFT, MemFlow et les autres ajoutent des modules inventés pour le flux : corrélation, warping, pyramide, itérations, upsampling convexe. FreeFlow les enlève tous : c'est un transformer encodeur–décodeur simple, **en une seule passe**.

<p align="center">
  <img src="documentation/slides/30_freeflow_contexte.png" width="720" alt="Les biais inductifs du flux optique">
</p>

### L'architecture, matrice par matrice

On suit une paire d'images du début à la fin. L'image devient une matrice **N × D**, cette taille ne change plus, et on revient aux pixels à la fin.

$$
N = \frac{H}{8} \cdot \frac{W}{8}, \qquad D \in \lbrace 256,\ 384,\ 512 \rbrace\ \text{(S, M, L)}
$$

<p align="center">
  <img src="documentation/figures/FreeFlow/freeflow_architecture_papier.png" width="720" alt="Architecture de FreeFlow (figure du papier)">
  <br><sub>Figure du papier (Bargatin et al., 2026).</sub>
</p>

| Étape | Ce qu'elle fait | Taille |
|---|---|---|
| 1. Patchification | l'image est coupée en patchs de 8 × 8, chaque patch devient une ligne de D nombres : xᵢ = Wₑ · vec(pᵢ) + b | 3 × H × W → N × D |
| 2. Win | attention dans 16 fenêtres fixes (détail local) | N × D |
| 3. Le bloc | Norm → Q, K, V → A = Softmax(log n/√64 · QKᵀ) → AV, têtes, résiduel, MLP | n × D |
| 4. Swin | les mêmes fenêtres, décalées d'une demi-fenêtre (on traverse les frontières) | N × D |
| 5. Global | attention sur toute l'image, sur une carte réduite de moitié (N/4 tokens) | N × D |
| 6. Encodeur ×L | L couches Win → Swin → Global, les deux images avec les mêmes poids (siamois) | X₁, X₂ → F¹, F² |
| 7. Décodeur | attention croisée : Q vient de l'image 1, K et V de l'image 2 | F¹, F² → Z |
| 8. Tête | 3 convolutions, retour aux pixels : f = 8 · U[0:2] | Z → H × W × 2 |

Le coût de chaque type d'attention est le même, alors que Global voit toute l'image :

$$
\text{Win, Swin : } 16 \times \left(\tfrac{N}{16}\right)^2 = \tfrac{N^2}{16}, \qquad \text{Global : } \left(\tfrac{N}{4}\right)^2 = \tfrac{N^2}{16}
$$

<details open>
<summary><b>Les 8 étapes en images</b></summary>

<p align="center">
  <img src="documentation/slides/31_freeflow_chemin.png" width="49%" alt="Le chemin de la matrice">
  <img src="documentation/slides/32_freeflow_patchs.png" width="49%" alt="Patchification">
</p>
<p align="center">
  <img src="documentation/slides/33_freeflow_win.png" width="49%" alt="Win">
  <img src="documentation/slides/34_freeflow_bloc_1.png" width="49%" alt="Le bloc (1/2)">
</p>
<p align="center">
  <img src="documentation/slides/35_freeflow_bloc_2.png" width="49%" alt="Le bloc (2/2)">
  <img src="documentation/slides/36_freeflow_swin.png" width="49%" alt="Swin">
</p>
<p align="center">
  <img src="documentation/slides/37_freeflow_global.png" width="49%" alt="Global">
  <img src="documentation/slides/38_freeflow_encodeur.png" width="49%" alt="Encodeur complet">
</p>
<p align="center">
  <img src="documentation/slides/39_freeflow_decodeur.png" width="49%" alt="Décodeur">
  <img src="documentation/slides/40_freeflow_tete.png" width="49%" alt="Tête de flux">
</p>
</details>

### Entraînement et données

1. **Pré-entraînement CroCo** : on cache 95 % de l'image 1, et le modèle la reconstruit en regardant l'image 2. Il apprend ainsi à retrouver les mêmes zones dans deux images, sans vérité terrain (3,7 M paires de photos réelles).
2. **Fine-tuning** sur des données où le vrai flux est connu : d'abord un mélange de 5 datasets (TartanAir, FlyingThings3D, Sintel, KITTI, HD1K), puis une version par benchmark (Sintel, KITTI, Spring).

<p align="center">
  <img src="documentation/slides/41_freeflow_entrainement.png" width="49%" alt="Entraînement de FreeFlow">
  <img src="documentation/slides/42_freeflow_datasets.png" width="49%" alt="Datasets de FreeFlow">
</p>

### Résultats publiés

- **Sintel** : 1er en Clean (EPE 0,68 px, −14 % par rapport à GeoViT) et en Final (1,48 px, −10 % par rapport à VideoFlow-MOF, qui utilise 5 images).
- **KITTI-2015** : 3e (Fl-all 3,23 %), derrière deux méthodes à plusieurs images, et 1er parmi les méthodes à 2 images.
- **Spring 1080p** : 1er en EPE (0,278), Fl (1,048) et WAUC (95,235), avec **8 fois moins de mémoire** que WAFT-DAv2 (2,58 Go contre 20,58 Go).

<p align="center">
  <img src="documentation/slides/44_freeflow_sintel_kitti.png" width="49%" alt="Sintel et KITTI">
  <img src="documentation/slides/45_freeflow_spring.png" width="49%" alt="Spring 1080p">
</p>

<p align="center">
  <img src="documentation/figures/FreeFlow/freeflow_carte_erreur_papier.png" width="720" alt="Carte d'erreur sur Spring (figure du papier)">
  <br><sub>Carte d'erreur sur Spring (figure du papier) : bleu = presque parfait, rouge = très faux.</sub>
</p>

### FreeFlow sur notre vidéo

| Mesure (GPU Tesla T4) | Valeur |
|---|---|
| Modèle | FreeFlow-L, checkpoint TaTSKH-HQ, fp16 |
| Vidéo | 576 × 1024 (portrait), 30 fps, 212 images → 211 paires |
| Tailles mesurées | X, F, Z : N × D = 9 216 × 512 (identique au papier) |
| Temps par paire | 573 ms (entre environ 530 et 600 ms) |
| Pic mémoire GPU | 1 367 Mo |
| Paramètres | 230,7 M |
| Erreur photométrique | 0,49 avec le flux, 1,20 sans (−59 %) |
| Middlebury | EPE 0,308 px · Fl-all 1,12 % |

<p align="center">
  <img src="documentation/figures/FreeFlow/freeflow_resultat.png" width="820" alt="Flux de FreeFlow sur notre vidéo">
</p>

- Le fond est blanc dans le flux et noir dans la magnitude : il ne bouge pas.
- La voiture est d'une seule couleur, avec une magnitude uniforme d'environ 5 px : elle avance d'un bloc, vers le haut. Les contours sont nets, jusqu'aux rétroviseurs.
- Exemple : le pixel (360, 568) a u = 0,4 et v = −5,5. Il se retrouve en (360, 563) dans l'image suivante.

<p align="center">
  <img src="documentation/figures/FreeFlow/freeflow_courbes.png" width="820" alt="Courbes au fil de la vidéo">
</p>

- **Temps** : il augmente un peu au fil de la vidéo (de 530 à 600 ms), alors que le calcul est toujours le même : c'est le GPU qui chauffe.
- **Mouvement moyen** : proche de 0 au début (voiture arrêtée), avec des pics vers les paires 140 à 190 (la voiture accélère), puis de nouveau 0 quand elle sort de l'image.
- **Erreur photométrique** : la courbe avec FreeFlow (orange) reste presque toujours sous la courbe sans flux (bleu). Les pics correspondent aux moments rapides : flou de mouvement et zones qui apparaissent derrière la voiture.

<p align="center">
  <img src="documentation/figures/FreeFlow/freeflow_video_extrait.png" width="420" alt="Extrait de la vidéo résultat">
  <br><sub>Extrait de la vidéo résultat : à gauche l'image, à droite le flux.</sub>
</p>

<p align="center">
  <img src="documentation/slides/61_freeflow_quand.png" width="720" alt="Quand utiliser FreeFlow">
</p>

---

## 5. Comparaison des trois modèles

### Résultats publiés (papiers)

| Critère | RAFT | MemFlow | FreeFlow-L |
|---|---|---|---|
| Images utilisées | 2 | 2 + mémoire | 2 |
| Itérations | 12 à 32 | 15 | aucune |
| Paramètres | 5,26 M | 6,27 M | 230,7 M |
| Mémoire 1080p | 7,97 Go | 8,06 Go | **2,58 Go** |
| Temps 1080p | **406 ms** | 754 ms | 607 ms |
| Spring EPE | 1,476 | 0,471 | **0,278** |
| Sintel Final (test) | 2,86 | 1,91 | **1,48** |

### Sur notre vidéo (Tesla T4, 576 × 1024)

| Mesure | RAFT (20 it.) | MemFlow (15 it.) | FreeFlow-L |
|---|---|---|---|
| Temps par paire | 705 ms | 1 478 ms | **573 ms** |
| Pic mémoire GPU | **726 Mo** | 3 794 Mo | 1 367 Mo |
| Baisse de l'erreur photométrique | −59 % | **−61 %** | −59 % |
| Middlebury EPE | 0,28 px | **0,214 px** | 0,308 px |
| Middlebury Fl-all | 0,94 % | **0,518 %** | 1,12 % |

**Ce qu'on en retient :**
- **FreeFlow** est le plus précis sur les benchmarks du papier, et il utilise 3 fois moins de mémoire en 1080p, alors qu'il a 44 fois plus de paramètres que RAFT. Il ne stocke jamais de grande matrice de corrélation et n'a pas d'itérations. Sur notre vidéo, il est le plus rapide.
- **Sur notre vidéo et sur Middlebury, les résultats sont plus serrés** : les trois modèles expliquent environ 60 % de l'erreur photométrique, et MemFlow obtient la meilleure EPE sur Middlebury. À notre résolution, plus petite que le 1080p, RAFT utilise moins de mémoire que FreeFlow. Middlebury contient de petits mouvements, alors que FreeFlow est conçu pour la haute résolution.
- **Le choix dépend du besoin** : RAFT pour un modèle léger et facile à réentraîner, MemFlow pour une vidéo continue (cohérence dans le temps), FreeFlow pour la haute résolution, les détails fins et une faible mémoire GPU.

<p align="center">
  <img src="documentation/slides/60_comparaison.png" width="720" alt="Comparaison RAFT, MemFlow, FreeFlow">
</p>

---

## 6. Reproduire nos résultats

Tous les tests ont été faits **en inférence seule**, avec un **GPU Tesla T4** : sur **Google Colab** pour RAFT et MemFlow, sur **Kaggle** pour FreeFlow. Sans GPU, le calcul se fait sur le processeur et devient très lent.

**RAFT** (Google Colab) :

1. Ouvrez [`notebooks/01_RAFT.ipynb`](notebooks/01_RAFT.ipynb) dans Colab et choisissez *Exécution → Modifier le type d'exécution → GPU T4*.
2. Lancez toutes les cellules (*Exécution → Tout exécuter*). Le code officiel et les poids `raft-sintel` sont téléchargés automatiquement.
3. À la première exécution, importez la vidéo [`vidéo de référence/video voiture 7s.mp4`](vidéo%20de%20référence/) quand le notebook la demande. Elle est ensuite réutilisée depuis le Drive.
4. Les figures, la vidéo résultat et `results.json` sont enregistrés dans `MyDrive/flow_optique/raft/`. Les figures de notre exécution sont aussi dans [`documentation/figures/RAFT/`](documentation/figures/RAFT/).

**FreeFlow** (Kaggle) : activez le GPU dans *Settings → Accelerator → GPU T4*, puis :

1. Créez un notebook Kaggle et ajoutez la vidéo comme dataset (*Add Input*).
2. Ouvrez le notebook du modèle, dans le dossier [`notebooks/`](notebooks/).
3. Dans la cellule de réglages, laissez `VIDEO_PATH = ""` (le notebook prend le premier `.mp4` trouvé) ou donnez le chemin du fichier.
4. Lancez toutes les cellules. Les figures et la vidéo résultat sont enregistrées dans `/kaggle/working/`.

Code et poids officiels de FreeFlow :

```python
!git clone -q https://github.com/msu-video-group/freeflow.git
!pip install -q ./freeflow

import torch
from freeflow import FreeFlow

model = FreeFlow.from_pretrained("a-yakovenko/optical-flow-FreeFlow-L-TaTSKH-HQ").eval().cuda()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    flow = model(pair)   # pair : [1, 2, 3, H, W], valeurs dans [0, 255]  ->  flow : [1, 2, H, W]
```

On utilise `float16`, car la T4 n'a pas de `bfloat16` matériel.

**MemFlow** : code officiel des auteurs (voir [Références](#8-références)), avec le checkpoint `MemFlowNet_sintel`.

---

## 7. Contenu du dépôt

```
.
├── README.md
├── notebooks/
│   ├── 01_RAFT.ipynb                 # RAFT sur notre vidéo + Middlebury (Colab)
│   ├── 02_MemFlow.ipynb              # MemFlow sur notre vidéo + Middlebury (Colab)
│   ├── 03_FreeFlow.ipynb             # FreeFlow sur notre vidéo + Middlebury (Kaggle)
│   └── 03_FreeFlow_inference.ipynb
├── vidéo de référence/
│   └── video voiture 7s.mp4          # vidéo testée (576 × 1024, 30 fps, 7 s)
└── documentation/
    ├── slides/                       # slides de la présentation, en images
    ├── figures/                      # figures de résultats
    │   ├── RAFT/
    │   ├── MemFlow/
    │   └── FreeFlow/
    ├── raft_result.mp4               # vidéo résultat RAFT (image | flux)
    └── memflow_result.mp4            # vidéo résultat MemFlow
```

---

## 8. Références

- Z. Teed, J. Deng. *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*. ECCV 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039) · [code](https://github.com/princeton-vl/RAFT)
- Q. Dong, Y. Fu. *MemFlow: Optical Flow Estimation and Prediction with Memory*. CVPR 2024. [arXiv:2404.04808](https://arxiv.org/abs/2404.04808) · [code](https://github.com/DQiaole/MemFlow)
- V. Bargatin, A. Yakovenko, K. Abud, D. Vatolin. *FreeFlow: A Bias-free Hierarchical Transformer for Optical Flow Estimation*. ECCV 2026. [arXiv:2609.11486](https://arxiv.org/abs/2609.11486) · [code](https://github.com/msu-video-group/freeflow)
- Middlebury Optical Flow : [vision.middlebury.edu/flow](https://vision.middlebury.edu/flow/)

Les figures marquées « figure du papier » appartiennent à leurs auteurs et sont reproduites à titre de citation. FreeFlow est distribué sous licence non commerciale (CC BY-NC-SA 4.0) : vérifiez la licence de chaque modèle avant toute réutilisation.
