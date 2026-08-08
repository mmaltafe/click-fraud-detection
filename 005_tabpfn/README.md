# 005_tabpfn

This directory contains the TabPFN adaptation experiments used before the federated LightGBM stage.

`000_fine_tune_head.py` reuses the best standard TabPFN grid-search configuration from `results/grid_search/tabpfn/{DATASET}`, keeps the pretrained TabPFN model frozen, extracts TabPFN embeddings, and trains a lightweight PyTorch head on top of them.

`001_fine_tune_head_staged_grid.py` runs a two-stage search: a compact first stage selects between linear and MLP heads, and a refined second stage explores the selected head family.

`002_threshold_calibration.py` evaluates whether campaign-level threshold calibration or a lightweight logistic regression layer over frozen TabPFN embeddings improves the binary attack/legitimate separation.

`003_stacking_meta_classifier.py` uses TabPFN probabilities or embeddings as inputs to classical meta-classifiers, including LightGBM, CatBoost, and logistic regression.

`004_embeddings_lightgbm_grid_search.py` runs the Bayesian optimization stage for TabPFN embeddings with a LightGBM meta-classifier. This is the main local reference used before the federated LightGBM experiments.
