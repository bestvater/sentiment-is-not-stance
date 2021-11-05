#!/usr/bin/env python
# coding: utf-8

####################################################################################################################
# Bestvater & Monroe; Sentiment != Stance
# 1_classifiers.py
# Python script to train various classifiers for sentiment and stance identification
# Script will save all results in ../data
####################################################################################################################

# SETUP

import pandas as pd
import numpy as np

import string
import re
import gc
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from simpletransformers.classification import ClassificationModel

import torch
print("Cuda available" if torch.cuda.is_available() is True else "CPU")
print("PyTorch version: ", torch.__version__)

punct = string.punctuation
punct = re.sub('#', '', punct)
punct = re.sub('@', '', punct)


####################################################################################################################
# WOMEN'S MARCH REPLICATION/EXTENSION

# Train BERT stance classifier

# read in data
balanced = pd.read_csv('./../data/WM_tweets_groundtruth.csv')
balanced = balanced[balanced['balanced_train']==1]
balanced = balanced[['text', 'stance']].reset_index(drop = True)


# Split into training and test sets
random.seed(101)
test_ind = random.choices(range(0, len(balanced)), k = 500)
train_ind = [i for i in range(0, len(balanced)) if i not in test_ind]

test = balanced.loc[test_ind]
train = balanced.loc[train_ind]

# Define parameters for BERT model
args = {
       'output_dir': 'outputs/',
       'cache_dir': 'cache/',

       'fp16': False,
       'fp16_opt_level': 'O1',
       'max_seq_length': 256,
       'train_batch_size': 16,
       'eval_batch_size': 16,
       'gradient_accumulation_steps': 1,
       'num_train_epochs': 5, 
       'weight_decay': 0,
       'learning_rate': 4e-5,
       'adam_epsilon': 1e-8,
       'warmup_ratio': 0.06,
       'warmup_steps': 0,
       'max_grad_norm': 1.0,

       'logging_steps': 50,
       'evaluate_during_training': False,
       'save_steps': 2000,
       'eval_all_checkpoints': True,
       'use_tensorboard': True,

       'overwrite_output_dir': True,
       'reprocess_input_data': True,
    }


# Load model
model = ClassificationModel('bert', 
                            'bert-base-cased', 
                            args = args, 
                            use_cuda= torch.cuda.is_available())

# Train model
model.train_model(train)


# Make predictions
result, model_outputs, wrong_predictions = model.eval_model(test)


print(f1_score(test['stance'], np.argmax(model_outputs, axis = 1)))




wm_corpus = pd.read_csv('./../data/FelmleeEtAl_corpus.csv')
eval_df = wm_corpus[['tweets', 'sentiment_untargeted']]
eval_df['sentiment_untargeted'] = 0
eval_df.columns = ['text', 'labels']
eval_df.head()





result, model_outputs, wrong_predictions = model.eval_model(eval_df)





def logit2prob(logit):
    odds = np.exp(logit)
    prob = odds / (1+odds)
    return(prob)





probs = []
logits = list(model_outputs[:,1])
for l in logits:
    probs.append(logit2prob(l))





wm_corpus['bert_stance'] = np.argmax(model_outputs, axis = 1)
wm_corpus['bert_probs'] = probs
wm_corpus.to_csv('./../data/FelmleeEtAl_corpus.csv', index = False)


# ### Produce measures for downstream regression analysis




analysis = pd.read_csv('./../data/WM_tweets_analysis_tweetscores.csv', encoding = 'utf-8')
analysis = analysis.sample(frac=1).reset_index(drop=True)

print(analysis.shape)

analysis.head()





scorer = SentimentIntensityAnalyzer()
vader_scores = [scorer.polarity_scores(tweet)['compound'] for tweet in list(analysis['text'])]
vader_scores_binary = [] 
for score in vader_scores:
    if score < 0:
        vader_scores_binary.append(0)
    elif score > 0:
        vader_scores_binary.append(1)
    else:
        vader_scores_binary.append(np.NaN)

analysis['vader_sentiment'] = vader_scores_binary
analysis['vader_sentiment_raw'] = vader_scores





result, model_outputs = model.predict(list(analysis['text']))





analysis['bert_stance'] = np.argmax(model_outputs, axis = 1)





analysis.to_csv('./../data/WM_tweets_analysis_tweetscores.csv', index=False)





del model
del result
del model_outputs
del wrong_predictions
gc.collect()
torch.cuda.empty_cache()

###################################################################################################################
# MOOD OF THE NATION EXAMPLE

# Compare Classifiers



# read in data
MOTN = pd.read_csv('./../data/MOTN_responses_groundtruth.csv', encoding = 'utf-8')
MOTN = MOTN.dropna(subset = ['edits_clean_text','trump_stance_auto']).reset_index(drop = True)

# split into folds for 5-fold crossvalidation
random.seed(101)
MOTN['fold'] = random.choices(range(1,6), k = len(MOTN))
MOTN = MOTN.sort_values(by = 'fold')
MOTN.head()


# Iterate through folds, train & test
vader_sentiment = []
SVM_sentiment = []
BERT_sentiment = []
SVM_stance = []
BERT_stance = []

for i in range(1,6):
    print('FOLD '+ str(i) + ' of 5')
    train = MOTN[MOTN['fold'] != i]
    test = MOTN[MOTN['fold'] == i]
    
    test = test[['edits_clean_text', 'qpos', 'trump_stance_auto']]
    train = train[['edits_clean_text', 'qpos', 'trump_stance_auto']]
    
    test.columns = ['text', 'sentiment', 'stance']
    train.columns = ['text', 'sentiment', 'stance']
    
    #######################################################################################################
    #Vader Sentiment Scorer
    print('training VADER sentiment scorer')
    scorer = SentimentIntensityAnalyzer()
    vader_scores = [scorer.polarity_scores(doc)['compound'] for doc in list(test['text'])]
    vader_scores_binary = [] 
    for score in vader_scores:
        if score < 0:
            vader_scores_binary.append(0)
        elif score > 0:
            vader_scores_binary.append(1)
        else:
            vader_scores_binary.append(np.NaN)

    vader_sentiment.extend(vader_scores_binary)
    
    #######################################################################################################
    # TFIDF-SVM Sentiment Classifier
    print('training SVM sentiment classifier')
    train_processed = train['text'].str.lower()
    train_processed = train['text'].str.replace('[{}]'.format(punct), '')
    test_processed = test['text'].str.lower()
    test_processed = test['text'].str.replace('[{}]'.format(punct), '')

    vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,1), max_features = 3000)

    X_train = vectorizer.fit_transform(train_processed)

    vocab = vectorizer.vocabulary_
    vectorizer = TfidfVectorizer(encoding = 'utf-8', vocabulary = vocab)

    X_test = vectorizer.fit_transform(test_processed)
    y_train = train['sentiment']
    y_test = test['sentiment']
    
    SVM = LinearSVC()
    SVM.fit(X_train, y_train)
    
    SVM_sentiment.extend(list(SVM.predict(X_test)))
    
    #######################################################################################################
    # BERT Sentiment Classifier
    print('training BERT sentiment classifier')
    train_df = train[['text', 'sentiment']]
    eval_df = test[['text', 'sentiment']]
    train_df['text'] = train_df['text'].str.lower()
    eval_df['text'] = eval_df['text'].str.lower()
    
    model = ClassificationModel('bert', 
                                'bert-base-cased', 
                                args = args, 
                                use_cuda= torch.cuda.is_available())

    model.train_model(train_df)
    
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    
    BERT_sentiment.extend(list(np.argmax(model_outputs, axis = 1)))
    
    del model
    del result
    del model_outputs
    del wrong_predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    #######################################################################################################
    # TFIDF-SVM Stance Classifier
    print('training SVM stance classifier')
    train_processed = train['text'].str.lower()
    train_processed = train['text'].str.replace('[{}]'.format(punct), '')
    test_processed = test['text'].str.lower()
    test_processed = test['text'].str.replace('[{}]'.format(punct), '')

    vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,1), max_features = 3000)

    X_train = vectorizer.fit_transform(train_processed)

    vocab = vectorizer.vocabulary_
    vectorizer = TfidfVectorizer(encoding = 'utf-8', vocabulary = vocab)

    X_test = vectorizer.fit_transform(test_processed)
    y_train = train['stance']
    y_test = test['stance']
    
    SVM = LinearSVC()
    SVM.fit(X_train, y_train)
    
    SVM_stance.extend(list(SVM.predict(X_test)))
    
    #######################################################################################################
    # BERT Stance Classifier
    print('training BERT stance classifier')
    train_df = train[['text', 'stance']]
    eval_df = test[['text', 'stance']]
    
    model = ClassificationModel('bert', 
                                'bert-base-cased', 
                                args = args, 
                                use_cuda= torch.cuda.is_available())

    model.train_model(train_df)
    
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    
    BERT_stance.extend(list(np.argmax(model_outputs, axis = 1)))
    
    del model
    del result
    del model_outputs
    del wrong_predictions
    gc.collect()
    torch.cuda.empty_cache()
    
MOTN['vader_sentiment'] = vader_sentiment
MOTN['SVM_sentiment'] = SVM_sentiment
MOTN['BERT_sentiment'] = BERT_sentiment
MOTN['SVM_stance'] = SVM_stance
MOTN['BERT_stance'] = BERT_stance


# Save predictions
MOTN = MOTN.sort_index()
MOTN.to_csv('./../data/MOTN_responses_groundtruth.csv', index = False)




####################################################################################################################
# Kavanaugh Tweets Example

# Compare Classifiers



# read in data, split into folds
KAV = pd.read_csv('./../data/kavanaugh_tweets_groundtruth.csv')
random.seed(101)
KAV['fold'] = random.choices(range(1,6), k = len(KAV))
KAV = KAV.sort_values(by = 'fold').dropna()
print(KAV.shape)
KAV.head()




# iterate through folds, train/test
vader_sentiment = []
SVM_sentiment = []
BERT_sentiment = []
SVM_stance = []
BERT_stance = []

for i in range(1,6):
    print('FOLD '+ str(i) + ' of 5')
    train = KAV[KAV['fold'] != i]
    test = KAV[KAV['fold'] == i]
    
    test = test[['text', 'sentiment', 'stance']]
    train = train[['text', 'sentiment', 'stance']]
    
    #######################################################################################################
    #Vader Sentiment Scorer
    print('training VADER sentiment scorer')
    scorer = SentimentIntensityAnalyzer()
    vader_scores = [scorer.polarity_scores(doc)['compound'] for doc in list(test['text'])]
    vader_scores_binary = [] 
    for score in vader_scores:
        if score < 0:
            vader_scores_binary.append(0)
        elif score > 0:
            vader_scores_binary.append(1)
        else:
            vader_scores_binary.append(np.NaN)

    vader_sentiment.extend(vader_scores_binary)
    
    #######################################################################################################
    # TFIDF-SVM Sentiment Classifier
    print('training SVM sentiment classifier')
    train_processed = train['text'].str.lower()
    train_processed = train['text'].str.replace('[{}]'.format(punct), '')
    test_processed = test['text'].str.lower()
    test_processed = test['text'].str.replace('[{}]'.format(punct), '')

    vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,1), max_features = 3000)

    X_train = vectorizer.fit_transform(train_processed)

    vocab = vectorizer.vocabulary_
    vectorizer = TfidfVectorizer(encoding = 'utf-8', vocabulary = vocab)

    X_test = vectorizer.fit_transform(test_processed)
    y_train = train['sentiment']
    y_test = test['sentiment']
    
    SVM = LinearSVC()
    SVM.fit(X_train, y_train)
    
    SVM_sentiment.extend(list(SVM.predict(X_test)))
    
    #######################################################################################################
    # BERT Sentiment Classifier
    print('training BERT sentiment classifier')
    train_df = train[['text', 'sentiment']]
    eval_df = test[['text', 'sentiment']]
    train_df['text'] = train_df['text'].str.lower()
    eval_df['text'] = eval_df['text'].str.lower()
    
    model = ClassificationModel('bert', 
                                'bert-base-cased', 
                                args = args, 
                                use_cuda= torch.cuda.is_available())

    model.train_model(train_df)
    
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    
    BERT_sentiment.extend(list(np.argmax(model_outputs, axis = 1)))
    
    del model
    del result
    del model_outputs
    del wrong_predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    #######################################################################################################
    # TFIDF-SVM Stance Classifier
    print('training SVM stance classifier')
    train_processed = train['text'].str.lower()
    train_processed = train['text'].str.replace('[{}]'.format(punct), '')
    test_processed = test['text'].str.lower()
    test_processed = test['text'].str.replace('[{}]'.format(punct), '')

    vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,1), max_features = 3000)

    X_train = vectorizer.fit_transform(train_processed)

    vocab = vectorizer.vocabulary_
    vectorizer = TfidfVectorizer(encoding = 'utf-8', vocabulary = vocab)

    X_test = vectorizer.fit_transform(test_processed)
    y_train = train['stance']
    y_test = test['stance']
    
    SVM = LinearSVC()
    SVM.fit(X_train, y_train)
    
    SVM_stance.extend(list(SVM.predict(X_test)))
    
    #######################################################################################################
    # BERT Stance Classifier
    print('training BERT stance classifier')
    train_df = train[['text', 'stance']]
    eval_df = test[['text', 'stance']]
    
    model = ClassificationModel('bert', 
                                'bert-base-cased', 
                                args = args, 
                                use_cuda= torch.cuda.is_available())

    model.train_model(train_df)
    
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)
    
    BERT_stance.extend(list(np.argmax(model_outputs, axis = 1)))
    
    del model
    del result
    del model_outputs
    del wrong_predictions
    gc.collect()
    torch.cuda.empty_cache()
    
KAV['vader_sentiment'] = vader_sentiment
KAV['SVM_sentiment'] = SVM_sentiment
KAV['BERT_sentiment'] = BERT_sentiment
KAV['SVM_stance'] = SVM_stance
KAV['BERT_stance'] = BERT_stance




# save preds
KAV = KAV.sort_index()
KAV.to_csv('./../data/kavanaugh_tweets_groundtruth.csv', index = False)



####################################################################################################################
#  Produce measures for downstream regression analysis

analysis = pd.read_csv('./../data/kavanaugh_tweets_analysis_tweetscores.csv')
analysis.head()





vader_sentiment = []
SVM_sentiment = []
BERT_sentiment = []
SVM_stance = []
BERT_stance = []


train = KAV
test = analysis

test = test[['text', 'sentiment', 'stance']]
train = train[['text', 'sentiment', 'stance']]

#######################################################################################################
#Vader Sentiment Scorer
print('training VADER sentiment scorer')
scorer = SentimentIntensityAnalyzer()
vader_scores = [scorer.polarity_scores(doc)['compound'] for doc in list(test['text'])]
vader_scores_binary = [] 
for score in vader_scores:
    if score < 0:
        vader_scores_binary.append(0)
    elif score > 0:
        vader_scores_binary.append(1)
    else:
        vader_scores_binary.append(np.NaN)

vader_sentiment.extend(vader_scores_binary)

#######################################################################################################
# TFIDF-SVM Sentiment Classifier
print('training SVM sentiment classifier')
train_processed = train['text'].str.lower()
train_processed = train['text'].str.replace('[{}]'.format(punct), '')
test_processed = test['text'].str.lower()
test_processed = test['text'].str.replace('[{}]'.format(punct), '')

vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,1), max_features = 3000)

X_train = vectorizer.fit_transform(train_processed)

vocab = vectorizer.vocabulary_
vectorizer = TfidfVectorizer(encoding = 'utf-8', vocabulary = vocab)

X_test = vectorizer.fit_transform(test_processed)
y_train = train['sentiment']
y_test = test['sentiment']

SVM = LinearSVC()
SVM.fit(X_train, y_train)

SVM_sentiment.extend(list(SVM.predict(X_test)))

#######################################################################################################
# BERT Sentiment Classifier
print('training BERT sentiment classifier')
train_df = train[['text', 'sentiment']]
eval_df = test[['text', 'sentiment']]
train_df['text'] = train_df['text'].str.lower()
eval_df['text'] = eval_df['text'].str.lower()

model = ClassificationModel('bert', 
                            'bert-base-cased', 
                            args = args, 
                            use_cuda= torch.cuda.is_available())

model.train_model(train_df)

result, model_outputs = model.predict(list(eval_df['text']))

BERT_sentiment.extend(list(np.argmax(model_outputs, axis = 1)))

del model
del result
del model_outputs
gc.collect()
torch.cuda.empty_cache()

#######################################################################################################
# TFIDF-SVM Stance Classifier
print('training SVM stance classifier')
train_processed = train['text'].str.lower()
train_processed = train['text'].str.replace('[{}]'.format(punct), '')
test_processed = test['text'].str.lower()
test_processed = test['text'].str.replace('[{}]'.format(punct), '')

vectorizer = TfidfVectorizer(encoding='utf-8', ngram_range=(1,1), max_features = 3000)

X_train = vectorizer.fit_transform(train_processed)

vocab = vectorizer.vocabulary_
vectorizer = TfidfVectorizer(encoding = 'utf-8', vocabulary = vocab)

X_test = vectorizer.fit_transform(test_processed)
y_train = train['stance']
y_test = test['stance']

SVM = LinearSVC()
SVM.fit(X_train, y_train)

SVM_stance.extend(list(SVM.predict(X_test)))

#######################################################################################################
# BERT Stance Classifier
print('training BERT stance classifier')
train_df = train[['text', 'stance']]
eval_df = test[['text', 'stance']]

model = ClassificationModel('bert', 
                            'bert-base-cased', 
                            args = args, 
                            use_cuda= torch.cuda.is_available())

model.train_model(train_df)

result, model_outputs = model.predict(list(eval_df['text']))

BERT_stance.extend(list(np.argmax(model_outputs, axis = 1)))

del model
del result
del model_outputs
gc.collect()
torch.cuda.empty_cache()
    
analysis['vader_sentiment'] = vader_sentiment
analysis['SVM_sentiment'] = SVM_sentiment
analysis['BERT_sentiment'] = BERT_sentiment
analysis['SVM_stance'] = SVM_stance
analysis['BERT_stance'] = BERT_stance





analysis.to_csv('./../data/kavanaugh_tweets_analysis_tweetscores.csv', index = False)

###########################################################################################################
# Add raw VADER scores to ground truth datasets (for appendix)

wm = pd.read_csv('./../data/WM_tweets_groundtruth.csv')
motn = pd.read_csv('./../data/MOTN_responses_groundtruth.csv', encoding = 'utf-8')
kav = pd.read_csv('./../data/kavanaugh_tweets_groundtruth.csv')

scorer = SentimentIntensityAnalyzer()

vader_scores = [scorer.polarity_scores(doc)['compound'] for doc in list(wm['text'])]
wm['vader_scores'] = vader_scores

vader_scores = [scorer.polarity_scores(doc)['compound'] for doc in list(motn['edits_clean_text'])]
motn['vader_scores'] = vader_scores

vader_scores = [scorer.polarity_scores(doc)['compound'] for doc in list(kav['text'])]
kav['vader_scores'] = vader_scores

wm.to_csv('./../data/WM_tweets_groundtruth.csv', index = False)
motn.to_csv('./../data/MOTN_responses_groundtruth.csv', index = False)
kav.to_csv('./../data/kavanaugh_tweets_groundtruth.csv', index = False) 