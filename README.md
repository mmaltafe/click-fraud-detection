# Article Reproducibility Scripts

This repository contains the scripts used to prepare the CFD campaign data and to reproduce the experimental results reported in the article. The code was copied from the research repository and organized as a standalone execution package.

The scripts preserve the original stage-based directory structure because several experiments load helper modules by path. The execution order below maps each directory to the corresponding article section and explains which outputs are expected.

For a compact script-by-script mapping, see `SCRIPT_INDEX.md`.

## Repository Layout

```text
./
├── 000_get_kaggle_data/              # Dataset download, campaign partitioning, and target preparation
├── 001_feature_extraction/           # Label Encoder, semantic headers, TF-IDF, and Sentence Transformer features
├── 002_feature_selection/            # PCA, TruncatedSVD, Chi2, and SelectKBest transformations
├── 003_machine_learning_evaluation/  # Initial classifiers, boosting models, TabPFN, and Flower MLP baselines
├── 004_grid_search/                  # Standard TabPFN grid-search stage
├── 005_tabpfn/                       # TabPFN adaptations and TabPFN embeddings + LightGBM Bayesian optimization
├── 006_federated_learning/           # Federated LightGBM, transfer matrix, aggregation strategies, and dev5new tests
├── utils/                            # Shared campaign, resume, target, and federated-LightGBM helpers
├── data/                             # Generated or downloaded data; not versioned except placeholders
├── models/                           # Optional local model cache
└── results/                          # Generated experiment outputs
```

## Article Mapping

The main article results are produced by the following stages:

| Article section | Purpose | Main scripts |
|---|---|---|
| Section 3 | Prepare the CFD campaign files and preserve the campaign-level structure | `000_get_kaggle_data/000_download_and_split_kaggle_data.py`, `000_get_kaggle_data/001_binarize_attack_labels.py` |
| Section 4 | Build feature representations and dimensionality-reduced variants | `001_feature_extraction/*.py`, `002_feature_selection/*.py` |
| Section 5.2 | Initial baseline comparison | `003_machine_learning_evaluation/000_all_classifiers.py`, `003_machine_learning_evaluation/001_boosting_algorithms.py`, `003_machine_learning_evaluation/002_tabpfn.py`, `003_machine_learning_evaluation/003_federated_mlp_flower.py` |
| Section 5.3 | TabPFN adaptations and TabPFN embeddings + LightGBM local reference | `005_tabpfn/002_threshold_calibration.py`, `005_tabpfn/003_stacking_meta_classifier.py`, `005_tabpfn/004_embeddings_lightgbm_grid_search.py` |
| Section 5.4 | Local, centralized, global federated, and local residual comparison | `006_federated_learning/001_local_federated_centralized_comparison.py`, `006_federated_learning/002_local_residual_federated_lightgbm.py`, `006_federated_learning/003_centralized_tabpfn_embeddings_lightgbm.py` |
| Section 5.5 | Cross-campaign transfer matrix | `006_federated_learning/004_cross_campaign_transfer_matrix.py` |
| Section 5.6 | Federated aggregation strategies | `006_federated_learning/005_transfer_weighted_federated_lightgbm.py`, `006_federated_learning/006_aggregation_strategies_lightgbm.py` |
| Section 5.7 | New low-data clients (`dev5new`) | `006_federated_learning/007_dev5new_low_data_transfer_models.py`, `006_federated_learning/008_dev5new_personalized_transfer_models.py` |

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The provided `.env` file contains safe defaults and no private tokens. If TabPFN authentication is required in your environment, set the token through your shell or a private local file that is not committed.

```bash
export TABPFN_TOKEN="your-token-if-required"
```

## Expected Data and Result Folders

The scripts write outputs relative to the repository root:

```text
data/raw/
data/processed/
data/tabpfn/
results/
```

`data/tabpfn/` is used to cache TabPFN embeddings so that federated LightGBM experiments can be rerun without recomputing the frozen TabPFN representation each time.

## Recommended Execution Order

Run commands from the `scripts_artigos` root.

### 1. Dataset Preparation

```bash
python 000_get_kaggle_data/000_download_and_split_kaggle_data.py
python 000_get_kaggle_data/001_binarize_attack_labels.py
```

The article focuses on the `dev5` subset and the `dev5new` low-data scenario. Both preserve campaigns as independent experimental units.

### 2. Feature Extraction

```bash
python 001_feature_extraction/000_label_encoder.py
python 001_feature_extraction/001_semantic_headers.py
python 001_feature_extraction/002_tf_idf.py
python 001_feature_extraction/003_sentence_transformer.py
```

### 3. Feature Selection and Dimensionality Reduction

```bash
python 002_feature_selection/000_pca.py
python 002_feature_selection/001_truncatedSVD.py
python 002_feature_selection/002_chi2.py
python 002_feature_selection/003_selectKBest.py
```

### 4. Initial Baselines

```bash
python 003_machine_learning_evaluation/000_all_classifiers.py
python 003_machine_learning_evaluation/001_boosting_algorithms.py
python 003_machine_learning_evaluation/002_tabpfn.py
python 003_machine_learning_evaluation/003_federated_mlp_flower.py
```

### 5. TabPFN Adaptations

```bash
python 004_grid_search/000_tabpfn_grid_search.py
python 005_tabpfn/000_fine_tune_head.py
python 005_tabpfn/001_fine_tune_head_staged_grid.py
python 005_tabpfn/002_threshold_calibration.py
python 005_tabpfn/003_stacking_meta_classifier.py
python 005_tabpfn/004_embeddings_lightgbm_grid_search.py
```

The final command produces the local TabPFN embeddings + LightGBM Bayesian-optimization reference used before the federated stage.

### 6. Federated LightGBM and Transfer-Based Aggregation

```bash
python 006_federated_learning/000_federated_lightgbm.py
python 006_federated_learning/001_local_federated_centralized_comparison.py
python 006_federated_learning/002_local_residual_federated_lightgbm.py
python 006_federated_learning/003_centralized_tabpfn_embeddings_lightgbm.py
python 006_federated_learning/004_cross_campaign_transfer_matrix.py
python 006_federated_learning/005_transfer_weighted_federated_lightgbm.py
python 006_federated_learning/006_aggregation_strategies_lightgbm.py
```

### 7. Low-Data New-Client Scenario

```bash
python 006_federated_learning/007_dev5new_low_data_transfer_models.py
python 006_federated_learning/008_dev5new_personalized_transfer_models.py
```

These scripts evaluate five new Facebook campaigns with the reverse five-fold protocol: one fold is used for training and the other four folds are used for testing.

## Notes on Reproducibility

- Some experiments are computationally expensive, especially TabPFN embeddings and Bayesian optimization.
- GPU execution is recommended for TabPFN-based stages.
- The scripts use fixed seeds where campaign sampling, folds, or model initialization require randomization.
- Results may vary slightly across hardware, Python versions, and library versions.
- Raw CFD files and generated results are intentionally not committed to this package.
