# READ ME: Replication Directory Overview
## Bestvater &amp; Monroe "Sentiment is Not Stance: Target-Aware Opinion Classification for Political Text Analysis"

|OVERVIEW              |                                        |
| ---------------------|----------------------------------------|
| Corresponding Author | Sam Bestvater; sam.bestvater@gmail.com |
| Co-Author            | Burt Monroe |
| Date                 | Nov 4, 2021 |
| Article              | Sentiment is Not Stance: Target-Aware Opinion Classification for Political Text Analysis  |
| Journal              | Political Analysis |
| Purpose              | This document explains how to replicate all results found in the paper and online supplementary materials, as well as to retrain all models used in the analysis. |


## Directory Structure
- requirements.txt : python packages required to run model training loop
- data/
  - FelmleeEtAl_corpus.csv : replication dataset for Felmlee Et Al 2020
  - intercoder_reliability_check.csv : multi-coded tweets for calculating IRR measures
  - kavanaugh_tweets_analysis_tweetscores.csv : tweets about the Kavanaugh hearings with Barbera (2015) ideology scores
  - kavanaugh_tweets_groundtruth.csv : tweets about the Kavanaugh hearings
  - MOTN_responses_groundtruth.csv : Mood of the Nation open-ended survey responses
  - WM_tweets_analysis_tweetscores.csv : tweets about the Women's March with Barbera (2015) ideology scores
  - WM_tweets_groundtruth.csv : tweets about the Women's March
- scripts/
  - 0_intercoder_reliability_check.ipynb : jupyter notebook to calculate IRR measures 
  - 1_train_classifiers.py : full training loop with 5-fold cross-validation for all VADER, SVM, & BERT measures in analysis
  - 2_train_additionalClassifiers.R : full training loop with 5-fold cross-validation for all lexicoder measures in analysis
  - 3_tweetscores.R : generate Barbera (2015) ideology scores
  - 4_analysisScript.R : core analysis, produces all results found in main text and appendix
- figures/ : all figures included in manuscript


## Instructions

### Option 1: You want to reproduce the results found in the paper
Follow the instructions in this section if you wish to reproduce the analysis found in the main text and appendix, without re-training classifiers or re-estimating ideology scores.

1. Run `4_analysis.R`

### Option 2: You want to replicate the results found in the paper by re-training the classifiers used
Follow the instructions in this section if you wish to replicate the analysis found in the main text and appendix using newly-trained classifiers. A few things are worth noting: first, a CUDA-enabeled GPU is highly recommended here. The code should run on a CPU instance, but it will take a *really* long time. Second, some of the results should be expected to differ slightly from those found in the published article, due to randomness in initializing the models at the training stage.

1. Install PyTorch for your system, following the instructions [here](https://pytorch.org/get-started/locally/)
2. Install the required python packages listed in `requirements.txt` by running `pip install -r requirements.txt` 
3. Run `1_train_classifiers.py`
4. Run `2_train_additionalClassifiers.R`
5. Run `4_analysis.R`

### Option 3: You want to replicate the results found in the paper by re-estimating the ideology scores used
Follow the instructions in this section if you wish to replicate the analysis found in the main text and appendix while re-estimating the ideology scores (from Barbera 2015).

1. Edit `3_tweetscores.R` to include your credentials for the Twitter API
2. Run `3_tweetscores.R`
3. Run `4_analysis.R`
