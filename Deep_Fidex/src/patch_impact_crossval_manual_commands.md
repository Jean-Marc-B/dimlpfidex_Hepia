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

## 3. Fonction de décroissance

La fonction de décroissance est maintenant un paramètre des commandes de règles :

```bash
--threshold_decay_function Linear
```

Valeurs disponibles :

```text
Linear
FastExponential
SlowExponential
FastPower
VeryFastPower
SlowPower
VerySlowPower
```

La valeur par défaut est `Linear`; il n'est plus nécessaire de modifier `fidexAlgo.cpp` ni de recompiler pour changer de décroissance.

## 4. Règles avec décroissance Linear

Ajouter explicitement `--threshold_decay_function Linear` pour tracer la configuration dans les commandes.

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
  --threshold_decay_function Linear \
  --fidelity_importance 1.0 \
  --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr00025_linear_fi1_thr1.out 2>&1 &
```

### CIFAR - fullFidex 0.95/0.95 FI=1, threshold_fidelity_only=1 -> FAIT

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

### CIFAR - fullFidex 0.95/0.95 FI=0.6, threshold_fidelity_only=0.6 -> FAIT

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

### CIFAR - earlyStopping 0.01 Linear FI=1 FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.0025 --threshold_decay_function Linear \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr00025_linear_fi1_thr1.out 2>&1 &

### CIFAR - earlyStopping 0.01 Linear FI=0.6, threshold_fidelity_only=0.6 FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_linear_fi06_thr06_V2 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.01 --threshold_decay_function Linear \
  --fidelity_importance 0.6 --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_linear_fi06_thr06_V2.out 2>&1 &

### CIFAR - earlyStopping 0.005 Linear FI=0.6, threshold_fidelity_only=0.6 FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr0005_linear_fi06_thr06 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.005 --threshold_decay_function Linear \
  --fidelity_importance 0.6 --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr0005_linear_fi06_thr06.out 2>&1 &

### CIFAR - earlyStopping 0.0025 Linear FI=0.6, threshold_fidelity_only=0.6 FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_linear_fi06_thr06 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.0025 --threshold_decay_function Linear \
  --fidelity_importance 0.6 --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr00025_linear_fi06_thr06.out 2>&1 &

### CIFAR - earlyStopping 0.00125 Linear FI=0.6, threshold_fidelity_only=0.6 FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr000125_linear_fi06_thr06 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.00125 --threshold_decay_function Linear \
  --fidelity_importance 0.6 --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr000125_linear_fi06_thr06.out 2>&1 &

### FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.01 --threshold_decay_function Linear \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_linear_fi1_thr1.out 2>&1 &

#FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr002_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.02 --threshold_decay_function Linear \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr002_linear_fi1_thr1.out 2>&1 &

#FAIT
nohup python -u crossVal.py --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr003_linear_fi1_thr1 \
  --rules --gpu -1 --fidexVersion fidexEarlyStopping --zeroFidelityRatio 0.03 --threshold_decay_function Linear \
  --fidelity_importance 1.0 --threshold_fidelity_only 1.0 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr003_linear_fi1_thr1.out 2>&1 &
```

## 5. Règles avec décroissance FastExponential

Utiliser `--threshold_decay_function FastExponential`.

### MNIST - earlyStopping 0.0025 FastExponential FI=0.6, threshold_fidelity_only=0.6 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_fastExp_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.0025 \
  --threshold_decay_function FastExponential \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr00025_fastExp_fi06_thr06.out 2>&1 &
```

### MNIST - earlyStopping 0.00125 FastExponential FI=0.6, threshold_fidelity_only=0.6 - FAIT

Permet de comparer FastPower vs FastExp sur le même ratio. Très utile pour savoir si FastExp est plus stable.

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr000125_fastExp_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.00125 \
  --threshold_decay_function FastExponential \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr000125_fastExp_fi06_thr06.out 2>&1 &
```

### CIFAR - earlyStopping 0.0025 FastExponential FI=0.6, threshold_fidelity_only=0.6 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr00025_fastExp_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.0025 \
  --threshold_decay_function FastExponential \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr00025_fastExp_fi06_thr06.out 2>&1 &
```

### CIFAR - earlyStopping 0.01 FastExponential FI=0.6, threshold_fidelity_only=0.6 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_fastExp_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --threshold_decay_function FastExponential \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_fastExp_fi06_thr06.out 2>&1 &
```

## 6. Règles avec décroissance SlowExponential

Utiliser `--threshold_decay_function SlowExponential`.

### CIFAR - earlyStopping 0.01 SlowExponential FI=0.6, threshold_fidelity_only=0.6 - FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_slowExp_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --threshold_decay_function SlowExponential \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_slowExp_fi06_thr06.out 2>&1 &
```

## 7. Règles avec décroissance FastPower

Utiliser `--threshold_decay_function FastPower`.

### MNIST - earlyStopping 0.00125 FastPower FI=0.6, threshold_fidelity_only=0.6 -> FAIT

Meilleur temps ponctuel, peu de règles, bon covering, très bon default. C’est celle qu’il faut valider.

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr000125_fastPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.00125 \
  --threshold_decay_function FastPower \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr000125_fastPower_fi06_thr06.out 2>&1 &
```

### CIFAR - earlyStopping 0.01 FastPower FI=0.6, threshold_fidelity_only=0.6 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_fastPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --threshold_decay_function FastPower \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_fastPower_fi06_thr06.out 2>&1 &
```

## 8. Règles avec décroissance VeryFastPower

Utiliser `--threshold_decay_function VeryFastPower`.

### MNIST - earlyStopping 0.00125 VeryFastPower FI=0.6, threshold_fidelity_only=0.6 - FAIT

Meilleure accuracy ponctuelle, mais potentiellement plus agressive. À lancer si le coût reste acceptable.

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Mnist --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr000125_veryFastPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.00125 \
  --threshold_decay_function VeryFastPower \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Mnist/logs_mnist_es_zfr000125_veryFastPower_fi06_thr06.out 2>&1 &
```

### CIFAR - earlyStopping 0.01 VeryFastPower FI=0.6, threshold_fidelity_only=0.6 - FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_veryFastPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --threshold_decay_function VeryFastPower \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_veryFastPower_fi06_thr06.out 2>&1 &
```

## 9. Règles avec décroissance SlowPower

Utiliser `--threshold_decay_function SlowPower`.

### CIFAR - earlyStopping 0.01 SlowPower FI=0.6, threshold_fidelity_only=0.6 -> FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_slowPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --threshold_decay_function SlowPower \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_slowPower_fi06_thr06.out 2>&1 &
```

## 10. Règles avec décroissance VerySlowPower

Utiliser `--threshold_decay_function VerySlowPower`.

### CIFAR - earlyStopping 0.01 VerySlowPower FI=0.6, threshold_fidelity_only=0.6 - FAIT

```bash
cd /home/HES/jeanmarc.boutay/dimlpfidex_Hepia/Deep_Fidex/src
nohup python -u crossVal.py \
  --n_folds 10 --crossval_seed 12 --python python \
  --dataset Cifar --statistic patch_impact_and_image --train_with_patches False \
  --folder_sufix _cv_es_zfr001_verySlowPower_fi06_thr06 \
  --rules --gpu -1 \
  --fidexVersion fidexEarlyStopping \
  --zeroFidelityRatio 0.01 \
  --threshold_decay_function VerySlowPower \
  --fidelity_importance 0.6 \
  --threshold_fidelity_only 0.6 \
  --alternative_folder "../../../CrossVal_patch_impact_and_image_cv_base_patch_impact/{fold_name}/files" \
  > ../../../data/Cifar/logs_cifar_es_zfr001_verySlowPower_fi06_thr06.out 2>&1 &
```

## 11. Résultats

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
