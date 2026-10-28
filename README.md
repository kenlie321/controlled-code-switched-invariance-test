# Controlled Code-Switched Invariance Test

Code, data and results for the paper **"Position and Language as Diagnostic Variables: A Controlled Invariance Test of Sentiment Classifiers on Indonesian-English Code-Switched Reviews"**

The study tests whether sentiment classifiers keep their prediction when one sentiment cue is **moved** (beginning, middle, end) or **translated** (Indonesian, English) inside the same code-switched review. Each of 272 reviews has 6 variants (3 positions x 2 languages), 1,632 variants in total. Six models are compared: Logistic Regression and Linear SVC (TF-IDF), BERT, mBERT, XLM-RoBERTa and IndoBERT.

## Repository contents

```
controlled-code-switched-invariance-test/
├── data/
│   └── final_accepted_dataset - Accepted Variants.csv   # 1,632 accepted variants
├── results/
│   ├── predictions_per_variant_test_split.csv           # predictions of all six models on the 330 test variants
│   ├── split_assignment_by_review.csv                   # train/test assignment of each review group (main split)
│   ├── repeated_split_result - repeated_splits_mean_std.csv   # Table 14 (mean and SD over five splits)
│   ├── repeated_split_result - repeated_splits_raw_results.csv # per-split results behind Table 14
│   ├── cz_token_stats.csv                # Table 15: training-set token statistics
│   ├── cz_lr_svc_coefficients.csv        # Logistic Regression and Linear SVC coefficients
│   ├── cz_intervention_rows.csv          # per-variant deletion and replacement results
│   ├── cz_intervention_summary.csv       # Table 16: summary of the intervention tests
│   ├── cz_margin_shift_summary.csv       # margin changes (token vs. random word vs. cue)
│   ├── cz_cue_deletion_summary.csv       # positive control: deleting the sentiment cue
│   ├── cz_attention_rows.csv             # per-sentence attention ratios
│   ├── cz_attention_summary.csv          # Table 17: attention summary
│   ├── cz_ig_rows.csv                    # per-sentence Integrated Gradients attributions
│   ├── cz_ig_summary.csv                 # Table 18: Integrated Gradients summary
│   └── cz_anchor_example.csv             # worked example for the review of Table 9
├── Position_and_Language_Code_Switched.ipynb   # all experiments (see "How to reproduce")
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Data

`data/final_accepted_dataset - Accepted Variants.csv` contains the 1,632 variants that passed the audit: review text, label and condition. It does **not** contain usernames, profile pictures or any account information. `reviewId` is a random UUID that groups the six variants of one review; it is not a Google Play identifier.

[List the real column names and what they mean, for example `reviewId`, the text column, `label`, `position`, `language`.]

The train/test assignment of the main split (`GroupShuffleSplit`, `random_state=42`) is in `results/split_assignment_by_review.csv`.

| | Review groups | Variants |
|---|---|---|
| Train | 217 | 1,302 |
| Test | 55 | 330 |
| Total | 272 | 1,632 (876 Positive, 756 Negative) |

**Source and license.** The base reviews come from the public Kaggle dataset [tiket-com-app-sentiment-analysis-reviews](https://www.kaggle.com/datasets/sasyablossom/tiket-com-app-sentiment-analysis-reviews) (Google Play reviews of the Tiket.com app). [State the license of that dataset and the license under which the variants are released.] The 72,000 original reviews are not redistributed here.

**Construction.** From the filtered reviews, 281 were selected with one dominant sentiment cue, the cue was translated and moved by the first author, and the variants were audited by one reviewer (naturalness >= 3, sentiment preserved, syntactically acceptable). Nine reviews were rejected, leaving 272. See Sections 2.1-2.4 of the paper.

## Setup

Experiments were run with Python 3.13.16, scikit-learn 1.6.1, PyTorch 2.11.0, Transformers 5.20.2 and Captum 0.9.0 (full list in `requirements.txt`).

```bash
git clone [REPO URL]
cd controlled-code-switched-invariance-test
pip install -r requirements.txt
```

Install PyTorch following [pytorch.org](https://pytorch.org) for your CUDA version if `pip install` does not give you a GPU build. The transformer experiments need a GPU. The notebook was run on [Google Colab, GPU TYPE]. [Add the approximate runtime.]

The notebook reads the dataset from [PATH USED IN THE NOTEBOOK]. Change that path if you run it from another location.

## How to reproduce

The notebook `Position_and_Language_Code_Switched.ipynb` contains every experiment in order. Its sections map to the paper as follows:

| Paper element | Notebook section | Output in `results/` |
|---|---|---|
| Tables 5-8 (overall, position, language, PCR/PFR) and error samples (Tables 9-13) | [SECTION NAME] | `predictions_per_variant_test_split.csv` |
| Table 14 (five repeated splits) | [SECTION NAME] | `repeated_split_result - *.csv` |
| Tables 15-18 (token statistics, deletion, attention, Integrated Gradients) | "Causal evidence" | `cz_*.csv` |

Settings (Section 2.6 of the paper):

- Linear models: TF-IDF with `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, no stopword removal. Logistic Regression `max_iter=1000, C=1.0, solver='lbfgs'`. Linear SVC `max_iter=1000, C=1.0, class_weight='balanced'`.
- Transformers: `bert-base-uncased`, `bert-base-multilingual-uncased`, `xlm-roberta-base`, `indolem/indobert-base-uncased`. `max_length=60`, `learning_rate=1e-5`, `batch_size=16`, 3 epochs, AdamW (weight decay 0.01), gradient clipping 1.0, no scheduler, no validation set, no tuning.
- Seed 42 for Python, NumPy and PyTorch. One fine-tuning run per model on the main split.
- Global Seed of 42 was set once at the start of the session. Run all cells in order

Fine-tuning on a GPU is not bit-for-bit deterministic, so rerunning may change a few predictions. The predictions behind the paper are stored in `results/predictions_per_variant_test_split.csv`. They cover the 330 test variants of the main split only.

## Metrics

- **Prediction Consistency Rate (PCR):** share of review groups whose six variants all receive the same prediction.
- **Prediction Flip Rate (PFR):** share of variants whose prediction differs from the original variant.

A high PCR with a low PFR means the manipulated variable did not affect the prediction. It does not mean the predictions are correct.

## Main findings

[Optional: two or three sentences copied from the abstract. State the scope: one dataset, one language pair, one split and one fine-tuning run per transformer.]

## Limitations

The variants were built and audited by a single person, the six background tokens were chosen after inspecting errors, the transformers were trained once without tuning, and all results come from one domain (Tiket.com reviews) and one language pair. See the Limitations paragraph of the paper.

## Citation

```bibtex
@article{lie[YEAR]position,
  title   = {Position and Language as Diagnostic Variables: A Controlled Invariance Test of Sentiment Classifiers on Indonesian-English Code-Switched Reviews},
  author  = {Lie, Kennedy and Santoso, Handri},
  journal = {[JOURNAL]},
  year    = {[YEAR]},
  doi     = {[DOI]}
}
```

## License

Code: MIT. See `LICENSE`.
