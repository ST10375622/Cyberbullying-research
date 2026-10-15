# Cyberbullying Detection – Research Code (RPDA8412)

Code for the study *"Natural Language Processing and Machine Learning Approaches for
Cyberbullying Detection on Social Media"* (L. Mosia, ST10375622).

It runs the whole methodology from the proposal on the two Kaggle datasets in `../Datasets`:

| Dataset | File | Size | Labels |
|---|---|---|---|
| Twitter Cyberbullying Classification | `cyberbullying_tweets.csv` | ~47k tweets | 6 classes: age, ethnicity, gender, religion, other_cyberbullying, not_cyberbullying |
| Jigsaw Toxic Comment Classification | `train.csv` | ~160k comments | multi-label: toxic, severe_toxic, obscene, threat, insult, identity_hate |

## Pipeline (matches the proposal methodology)

1. **Pre-processing** (`src/preprocessing.py`)
   - Strips URLs, @mentions and IP addresses (anonymisation for POPIA), special characters and extra whitespace, then lower-cases.
   - Removes duplicates by MD5 hash.
   - Identifies language with langdetect.
   - Tokenises with NLTK.
   - Removes stop-words using the NLTK list and lemmatises.
2. **Label harmonisation** (`src/harmonisation.py`): maps both datasets onto a common scheme, as described in the Limitations section. The scheme has four classes: **not / explicit / implicit cyberbullying / hate speech**. There is also a **binary** task. The Twitter set additionally keeps its **native 6 classes**.
3. **Split**: stratified **70 / 15 / 15** (train / validation / test), seed 42. The split is identical for every model.
4. **Models**. All are tuned or early-stopped on validation macro-F1 and tested once on the test split.

   | # | Model | Features | Imbalance handling |
   |---|---|---|---|
   | 1 | Logistic Regression | TF-IDF (1–2 grams) | SMOTE |
   | 2 | SVM (linear, calibrated) | TF-IDF | SMOTE |
   | 3 | Random Forest | TF-IDF | SMOTE |
   | 4 | BiLSTM (Keras) | learned embeddings | class weights |
   | 5 | BERT (`bert-base-uncased`) | WordPiece | class-weighted loss |
   | 6 | AfriBERTa (`castorini/afriberta_base`) | SentencePiece | class-weighted loss |

5. **Evaluation** (`src/evaluation.py`)
   - Metrics: accuracy, precision, recall, F1 (macro and weighted), AUC-ROC, bootstrap 95% CI of the F1, and confusion matrices.
   - Figures: ROC curves.
   - Deployment cost: inference time per post.
6. **Cross-dataset benchmarking**: each binary model trained on one dataset is also tested on the other dataset.
7. **Statistics** (`src/statistical_tests.py`): McNemar's test with Holm correction, paired bootstrap, the Friedman test, and two-proportion z-tests with Wilcoxon signed-rank tests for H1a, H1b and H1c.
8. **Analysis** (`src/error_analysis.py`): most predictive n-grams, misclassified posts, and performance by post length and by language.
9. **Report** (`src/report.py`): produces `outputs/findings_summary.md`, organised by research question, plus all tables and figures.
10. **Deployment framework** (`deployment/`): REST API and CLI using the selected model, with an allow / human review / auto-hide policy.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

## Running

```bash
python run_all.py --quick                      # smoke test on 3k rows per dataset (~5-10 min CPU)
python run_all.py --stages prepare eda classical bilstm stats analysis report   # local CPU
python run_all.py --stages transformers        # GPU recommended (Colab / Kaggle)
python run_all.py --stages stats analysis report   # rebuild statistics + findings after any training
python run_all.py                              # everything
```

Useful options:

- `--datasets twitter` runs one dataset only.
- `--models bert` runs a subset of models.
- `--max-rows 40000` takes an equal stratified subsample for *all* models.
- `--transformer-epochs 3` changes the number of fine-tuning epochs.
- `--no-langdetect` skips language identification.

### Google Colab (for BERT / AfriBERTa)

Use `notebooks/colab_runner.ipynb`:

1. Upload the `RPDA8412` folder (containing `Datasets/` and `Cyberbullying code/`) to Google Drive.
2. Select a GPU runtime.
3. Run all cells.

Each result is saved as its own file under `outputs/metrics` and `outputs/predictions`, so you can train on Colab and on a local machine and combine the results. To do that, copy the `outputs` folder back and run `--stages stats analysis report`.

Approximate run times on a T4 GPU, for the full data at 2 epochs: BERT and AfriBERTa take about 1.5–2.5 h in total. The classical models and BiLSTM on a laptop CPU take about 1–2 h. Random Forest with SMOTE on Jigsaw is the slowest part.

## Outputs (`outputs/`)

| Path | Contents | Research question |
|---|---|---|
| `findings_summary.md` | All key findings, organised by RQ and hypothesis | All |
| `tables/main_model_comparison.csv` | Every metric for each model × dataset × task | Main RQ, SQ2 |
| `figures/comparison_*.png`, `heatmap_f1_macro.png` | Model comparison charts | Main RQ, SQ2 |
| `figures/cm__*.png`, `roc__*.png` | Confusion matrices and ROC curves | Main RQ |
| `stats/pairwise_model_tests.csv`, `hypothesis_results.json` | Significance tests (H1a and H1b) | SQ2 |
| `stats/H1c_explicit_vs_implicit.csv`, `figures/per_class_f1__*.png` | Results per category | SQ3 |
| `tables/class_imbalance_ablation.csv`, `figures/class_imbalance_ablation.png` | Comparison of no rebalancing, class weights and SMOTE | SQ4 |
| `tables/eda_*.csv`, `eda/*.png` | Dataset characteristics, imbalance, lengths, languages, n-grams | SQ4 |
| `tables/cross_dataset_generalisation.csv` | Cross-dataset benchmarking | SQ4, contribution |
| `error_analysis/` | Feature importance, misclassified posts, F1 by length and by language | SQ3, SQ4 |
| `tables/deployment_candidates.csv`, `models/deployment_choice.json` | Model selected for deployment | SQ5 |

## Deployment demo

```bash
python deployment/predict.py "you are so stupid nobody likes you" "have a lovely day"
uvicorn app:app --app-dir deployment --port 8000      # then open http://localhost:8000/docs
```

## Notes and assumptions

- **Explicit vs implicit**: neither dataset labels this directly, so it is derived as follows. A cyberbullying post is *explicit* if it contains an overtly abusive or profane term from the editable `EXPLICIT_LEXICON` in `config.py`. For Jigsaw, a post is also explicit if it is labelled obscene, insult, threat or severe_toxic. Otherwise the post is *implicit*. Identity-targeted abuse is *hate speech*: Twitter's religion, ethnicity and gender categories and Jigsaw's identity_hate label. Report this as a limitation.
- **SMOTE** is applied only to the TF-IDF training split and is never applied to validation or test data. Sequence models (the BiLSTM and the transformers) use class-weighted loss instead, because SMOTE cannot create synthetic raw text.
- **Stop-words**: transformers receive the cleaned text without stop-word removal, because they need full context. The classical models and the BiLSTM receive the stop-word-free, lemmatised text.
- **Ethics**: `error_analysis/misclassified_examples_best_models.csv` contains offensive text for manual review, so limit your exposure. Usernames and IP addresses are removed during cleaning.
- `--quick` results only check that the pipeline runs. They are **not** research findings.
