# Cross-validation patch_impact_and_image - commandes manuelles

Cette procédure compare les configs de règles sur les mêmes 10 folds.
On entraîne donc une seule fois par dataset, puis chaque config `--rules` réutilise les fichiers du fold correspondant avec `--alternative_folder`.

## 1. Préparation

Depuis la racine du repo :

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia
source .venv/bin/activate
```



## 2. Lancer les 10 entraînements/folds de base

Ces commandes lancent `train + stats + second_train`, avec GPU, sans génération de règles.
Elles créent les fichiers réutilisés ensuite par toutes les configs.

### MNIST base

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 \
  --crossval_seed 12 \
  --python python \
  --dataset Mnist \
  --statistic patch_impact_and_image \
  --train_with_patches False \
  --folder_sufix _cv_base_patch_impact \
  --train --stats --second_train \
  --gpu 0 \
  > ../../../data/Mnist/logs_mnist_base_patch_impact.out 2>&1 &
```

### CIFAR base

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 \
  --crossval_seed 12 \
  --python python \
  --dataset Cifar \
  --statistic patch_impact_and_image \
  --train_with_patches False \
  --folder_sufix _cv_base_patch_impact \
  --train --stats --second_train \
  --gpu 1 \
  > ../../../data/Cifar/logs_cifar_base_patch_impact.out 2>&1 &
```

Attendre que ces deux commandes soient terminées avant de lancer les règles.

## 3. Fonction de décroissance à modifier et recompilation

La fonction de décroissance est dans :

```text
fidex/cpp/src/fidexAlgo.cpp
```

Modifier cette ligne :

```cpp
constexpr ThresholdDecayFunction kThresholdDecayFunction = ThresholdDecayFunction::FastExponential;
```

Valeurs utilisées ici :

```cpp
ThresholdDecayFunction::Linear
ThresholdDecayFunction::FastExponential
ThresholdDecayFunction::FastPower
ThresholdDecayFunction::SlowPower
```

Après chaque modification de cette ligne, recompiler :

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia
cmake --build .
```

Ensuite lancer les configs de règles correspondant à cette fonction de décroissance.

## 4. Règles avec décroissance Linear

Modifier `kThresholdDecayFunction` en :

```cpp
constexpr ThresholdDecayFunction kThresholdDecayFunction = ThresholdDecayFunction::Linear;
```

Puis recompiler avec les commandes de la section 3.

### MNIST - fullFidex 0.95/0.95 FI=1, threshold_fidelity_only=1 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_full_095_095_fi1_thr1 \
  --rules --gpu -1 \
  --fidexVersion fidexFull \
  --zeroFidelityRatio 1.0 \
  --fidelity_importance 1.0 \
  --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_full_fi1_thr1.out 2>&1 &
```

### MNIST - fullFidex 0.95/0.95 FI=0.6, threshold_fidelity_only=0.6 -> FAIT (logs dans DeepFidex)

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_full_095_095_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexFull \
  --zeroFidelityRatio 1.0 \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_full_fi06_thr06.out 2>&1 &
```

### MNIST - earlyStopping 0.0025 Linear FI=1, threshold_fidelity_only=1 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_linear_fi1_thr1 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.0025 \
  --fidelity_importance 1.0 \
  --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr00025_linear_fi1_thr1.out 2>&1 &
```

### CIFAR - fullFidex 0.95/0.95 FI=1, threshold_fidelity_only=1

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_full_095_095_fi1_thr1 \
  --rules --gpu -1 \
  --fidexVersion fidexFull \
  --zeroFidelityRatio 1.0 \
  --fidelity_importance 1.0 \
  --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_full_fi1_thr1.out 2>&1 &
```

### CIFAR - fullFidex 0.95/0.95 FI=0.6, threshold_fidelity_only=0.6

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_full_095_095_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexFull \
  --zeroFidelityRatio 1.0 \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_full_fi06_thr06.out 2>&1 &
```

### CIFAR - earlyStopping Linear

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src

 ### LANCE
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.0025 \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr00025_linear_fi1_thr1.out 2>&1 &

### LANCE
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_linear_fi06_thr06 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.01 \
  --fidelity_importance 0.6 --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_linear_fi06_thr06.out 2>&1 &

### LANCE
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.01 \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_linear_fi1_thr1.out 2>&1 &

nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr002_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.02 \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr002_linear_fi1_thr1.out 2>&1 &

nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr003_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.03 \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr003_linear_fi1_thr1.out 2>&1 &
```

## 5. Règles avec décroissance FastExponential

Modifier `kThresholdDecayFunction` en :

```cpp
constexpr ThresholdDecayFunction kThresholdDecayFunction = ThresholdDecayFunction::FastExponential;
```

Puis recompiler avec les commandes de la section 3.

### MNIST - earlyStopping 0.0025 FastExponential FI=0.6, threshold_fidelity_only=0.6

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_fastExp_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.0025 \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr00025_fastExp_fi06_thr06.out 2>&1 &
```

## 6. Règles avec décroissance FastPower

Modifier `kThresholdDecayFunction` en :

```cpp
constexpr ThresholdDecayFunction kThresholdDecayFunction = ThresholdDecayFunction::FastPower;
```

Puis recompiler avec les commandes de la section 3.

### MNIST - earlyStopping 0.00125 FastPower FI=0.6, threshold_fidelity_only=0.6

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr000125_fastPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.00125 \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr000125_fastPower_fi06_thr06.out 2>&1 &
```

## 7. Règles avec décroissance SlowPower

Modifier `kThresholdDecayFunction` en :

```cpp
constexpr ThresholdDecayFunction kThresholdDecayFunction = ThresholdDecayFunction::SlowPower;
```

Puis recompiler avec les commandes de la section 3.

### CIFAR - earlyStopping 0.01 SlowPower FI=0.6, threshold_fidelity_only=0.6

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_slowPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_slowPower_fi06_thr06.out 2>&1 &
```

## 8. Résultats

Chaque config écrit un résumé ici :

```text
/home/HES/jeanmarc.boutay/dimlpfidex_Hepia/data/Mnist/evaluation/ScanFull/CrossVal_*/crossval_stats.txt
/home/HES/jeanmarc.boutay/dimlpfidex_Hepia/data/Cifar/evaluation/ScanFull/CrossVal_*/crossval_stats.txt
```

Chaque dossier contient aussi :

```text
crossval_commands.txt
crossval_stats.json
```

Les fichiers lourds `train/stats/second_train` restent dans le dossier de base :

```text
CrossVal_patch_impact_and_image_cv_base_patch_impact/fold_XX/files
```

Les configs règles ne dupliquent pas ces fichiers ; elles les lisent via `--alternative_folder`.
