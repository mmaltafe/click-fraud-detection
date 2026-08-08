# Script Index by Article Section

The stage folders were preserved to keep path-based imports reproducible. The remaining scripts were renumbered after removing development-only analyses, and this index gives each script an article-oriented execution label.

| Order | Article section | Script | Output role |
|---:|---|---|---|
| 01 | Section 3 | `000_get_kaggle_data/000_download_and_split_kaggle_data.py` | Downloads and partitions the CFD dataset into campaign-level raw files. |
| 02 | Section 3 / Section 4 | `000_get_kaggle_data/001_binarize_attack_labels.py` | Creates the binary attack/legitimate target used in the experiments. |
| 03 | Section 4 | `001_feature_extraction/000_label_encoder.py` | Generates the label-encoded feature representation. |
| 04 | Section 4 | `001_feature_extraction/001_semantic_headers.py` | Generates semantic HTTP/header-derived features. |
| 05 | Section 4 | `001_feature_extraction/002_tf_idf.py` | Generates TF-IDF features from concatenated request fields. |
| 06 | Section 4 | `001_feature_extraction/003_sentence_transformer.py` | Generates dense Sentence Transformer features. |
| 07 | Section 4 | `002_feature_selection/000_pca.py` | Applies PCA to compatible feature matrices. |
| 08 | Section 4 | `002_feature_selection/001_truncatedSVD.py` | Applies TruncatedSVD to sparse or dense feature matrices. |
| 09 | Section 4 | `002_feature_selection/002_chi2.py` | Applies Chi-square feature selection. |
| 10 | Section 4 | `002_feature_selection/003_selectKBest.py` | Applies SelectKBest feature selection. |
| 11 | Section 5.2 | `003_machine_learning_evaluation/000_all_classifiers.py` | Evaluates the initial classical classifier baselines. |
| 12 | Section 5.2 | `003_machine_learning_evaluation/001_boosting_algorithms.py` | Evaluates LightGBM, XGBoost, and CatBoost baselines. |
| 13 | Section 5.2 | `003_machine_learning_evaluation/002_tabpfn.py` | Evaluates the standard TabPFN classifier. |
| 14 | Section 5.2 | `003_machine_learning_evaluation/003_federated_mlp_flower.py` | Evaluates the initial Flower MLP federated baseline. |
| 15 | Section 5.3 | `004_grid_search/000_tabpfn_grid_search.py` | Searches standard TabPFN configurations. |
| 16 | Section 5.3 | `005_tabpfn/000_fine_tune_head.py` | Trains a frozen-TabPFN feature head. |
| 17 | Section 5.3 | `005_tabpfn/001_fine_tune_head_staged_grid.py` | Runs the staged frozen-head search. |
| 18 | Section 5.3 | `005_tabpfn/002_threshold_calibration.py` | Evaluates TabPFN threshold calibration. |
| 19 | Section 5.3 | `005_tabpfn/003_stacking_meta_classifier.py` | Evaluates TabPFN stacking and meta-classifier variants. |
| 20 | Section 5.3 | `005_tabpfn/004_embeddings_lightgbm_grid_search.py` | Produces the TabPFN embeddings + LightGBM Bayesian-optimization reference. |
| 21 | Section 5.4 | `006_federated_learning/000_federated_lightgbm.py` | Runs the reference federated LightGBM over TabPFN embeddings. |
| 22 | Section 5.4 | `006_federated_learning/001_local_federated_centralized_comparison.py` | Compares local, centralized, and global federated settings. |
| 23 | Section 5.4 | `006_federated_learning/002_local_residual_federated_lightgbm.py` | Evaluates local residual personalization over the federated model. |
| 24 | Section 5.4 | `006_federated_learning/003_centralized_tabpfn_embeddings_lightgbm.py` | Evaluates centralized LightGBM over the shared TabPFN embedding representation. |
| 25 | Section 5.5 | `006_federated_learning/004_cross_campaign_transfer_matrix.py` | Computes the directed cross-campaign transfer matrix. |
| 26 | Section 5.6 | `006_federated_learning/005_transfer_weighted_federated_lightgbm.py` | Evaluates transfer-weighted federated aggregation. |
| 27 | Section 5.6 | `006_federated_learning/006_aggregation_strategies_lightgbm.py` | Evaluates adaptive, similarity-based, robust, and personalized aggregation strategies. |
| 28 | Section 5.7 | `006_federated_learning/007_dev5new_low_data_transfer_models.py` | Evaluates the low-data `dev5new` transfer scenario. |
| 29 | Section 5.7 | `006_federated_learning/008_dev5new_personalized_transfer_models.py` | Evaluates personalized transfer aggregation in the `dev5new` scenario. |
