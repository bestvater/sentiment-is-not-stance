####################################################################################################################
# Bestvater & Monroe; Sentiment != Stance
# 4_analysisScript.R
# R script containing core analysis and producing figures/tables
####################################################################################################################

# SETUP
require(ggplot2)
require(forcats)
require(dplyr)
require(stargazer)
require(MLmetrics)
require(irr)

theme_set(theme_bw())

PANLightBlue <- '#69B2DA'
PANDarkBlue <- '#137EC2'

report_results <- function(df, y_true_name, y_pred_name, NA_rule = 'random'){
  #function calculates F1 scores over N folds, with different rules for missing values
  folds <- unique(df$fold)
  metrics <- c()
  for(i in 1:length(folds)){
    fold <- folds[i]
    fold_df <- df[df$fold == fold, ]
    rownames(fold_df) <- 1:nrow(fold_df)
    y_true <- fold_df[, y_true_name]
    if(NA_rule == 'drop'){
      y_pred <- fold_df[, y_pred_name]
    }else if(NA_rule == 'random'){
      y_pred <- c()
      for(j in 1:nrow(fold_df)){
        pred <- fold_df[j, y_pred_name]
        if(is.na(pred)){
          y_pred <- c(y_pred, sample(unique(y_true), 1, replace = T))
        }else{
          y_pred <- c(y_pred, pred)
        }
      }
    }else if(NA_rule == 'strict'){
      y_pred <- c()
      for(j in 1:nrow(fold_df)){
        pred <- fold_df[j, y_pred_name]
        if(is.na(pred)){
          opposite <- ifelse(fold_df[j, y_true_name]==1, 0, 1)
          y_pred <- c(y_pred, opposite)
        }else{
          y_pred <- c(y_pred, pred)
        }
      }
    }else{
      print('Incorrect value supplied for NA rule. Defaulting to random')
      y_pred <- fold_df[, y_pred_name]
    }
    F1 <- MLmetrics::F1_Score(y_true, y_pred)
    metrics <- c(metrics, F1)
  }
  return(metrics)
}

random_fill_NA <- function(input_var){
  output_var <- c()
  for(i in 1:length(input_var)){
    obs <- input_var[i]
    if(is.na(obs)){
      output_var <- c(output_var, sample(unique(input_var), 1, replace = T))
    }else{
      output_var <- c(output_var, obs)
    }
  }
  return(output_var)
}

se <- function(x) sd(x)/sqrt(length(x))

####################################################################################################################
# WOMEN'S MARCH REPLICATION/EXTENSION

wm_corpus <- read.csv('./../data/FelmleeEtAl_corpus.csv', stringsAsFactors = F)

# produce figure 1
plot_df <- wm_corpus[wm_corpus$place %in% c('washington', 'nyc', 'boston', 'losangeles', 'chicago', 'denver'), ]
plot_df$sentiment <- ifelse(plot_df$sentiment_untargeted > 0, 'positive', NA)
plot_df$sentiment <- ifelse(plot_df$sentiment_untargeted < 0, 'negative', plot_df$sentiment)
plot_df$stance <- ifelse(plot_df$bert_stance == 1, 'positive', 'negative')
plot_df <- plot_df %>% 
  select(place, sentiment, stance, sentiment_untargeted) %>%
  mutate(sentiment = fct_relevel(sentiment, 'positive', 'negative'),
         place = fct_relevel(place, 'washington', 'nyc', 'boston', 'losangeles', 'chicago', 'denver'))

plot_df$place = factor(plot_df$place, 
                       levels = c('washington', 'nyc', 'boston', 'losangeles', 'chicago', 'denver'),
                       labels = c('Washington', 'New York City', 'Boston', 'Los Angeles', 'Chicago', 'Denver'))

ggplot(data = plot_df, aes(place, sentiment_untargeted))+
  geom_boxplot(fill = PANLightBlue)+
  labs(y = 'Sentiment', x = NULL)

ggsave('./../figures/F1.pdf', width = 8, height = 3, units = 'in', dpi = 600)


# function to produce sentiment/stance correlation tables
senti_stance_cor_table <- function(v1, v2, filename){
  t1a <- as.data.frame.matrix(table(v1, v2)) 
  names(t1a) <- c('Negative Sentiment', 'Positive Sentiment')
  rownames(t1a) <- c('Opposing Stance', 'Approving Stance')
  t1a <- rev(t1a)
  t1a <- t1a[nrow(t1a):1, ]
  n <- sum(table(v1, v2))
  rval <- cor.test(v1, v2)$estimate
  
  sink(filename)
  write(stargazer(t1a, summary = F, type = 'text'))
  cat('\n\n')
  cat(paste('N = ', n))
  cat('\n')
  cat(paste('r = ', round(rval, 2)))
  sink()
  
  system(paste('cat ', filename))
}



# produce counts for Table 1
wm_handcoded <- read.csv('./../data/WM_tweets_groundtruth.csv')

senti_stance_cor_table(wm_handcoded$stance, wm_handcoded$sentiment, './../tables/Table_1.txt')


# produce split counts for appendix (Table A7)
wm_handcoded$moderate_sent <- ifelse(wm_handcoded$vader_scores > 0.5, 0, 1)
wm_handcoded$moderate_sent <- ifelse(wm_handcoded$vader_scores < -0.5, 0, wm_handcoded$moderate_sent)

# moderate
senti_stance_cor_table(wm_handcoded$stance[wm_handcoded$moderate_sent == 1], 
                       wm_handcoded$sentiment[wm_handcoded$moderate_sent == 1],
                       './../tables/Table_A7_moderate.txt')

# extreme
senti_stance_cor_table(wm_handcoded$stance[wm_handcoded$moderate_sent == 0], 
                       wm_handcoded$sentiment[wm_handcoded$moderate_sent == 0],
                       './../tables/Table_A7_extreme.txt')


 # produce figure 2
plot_df <- plot_df[complete.cases(plot_df), ]

ggplot(data = plot_df, aes(sentiment, stance, group = place))+
  geom_jitter(alpha = 0.5, color = PANLightBlue)+
  facet_wrap(facets = vars(place), strip.position = 'bottom')+
  labs(x = NULL, y = NULL)+
  scale_x_discrete(position = "top", labels = c('Positive Sentiment', 'Negative Sentiment'))+
  scale_y_discrete(labels = c('Opposing Stance', 'Approving Stance'))+
  theme(axis.text.y = element_text(angle = 90, hjust = 0.5))

ggsave('./../figures/F2.pdf', width = 8, height = 6, units = 'in', dpi = 600)


# downstream regression; produce figure 3, table 2
analysis_df <- read.csv('./../data/WM_tweets_analysis_tweetscores.csv', stringsAsFactors = F)
analysis_df$ideology_score <- ifelse(analysis_df$ideology_score == Inf, 2.5, analysis_df$ideology_score)
analysis_df$ideology_score <- ifelse(analysis_df$ideology_score == -Inf, -2.5, analysis_df$ideology_score)

m1 <- glm(data = analysis_df, vader_sentiment~ideology_score, family = binomial(link = 'logit'))

m2 <- glm(data = analysis_df, bert_stance~ideology_score, family = binomial(link = 'logit'))

m_ref <- glm(data = analysis_df, stance~ideology_score, family = binomial(link = 'logit'))

# table 2
stargazer(m1, m2, m_ref, type = 'latex',
          star.char = c('','',''),
          notes = '',
          digits = 2,
          notes.append = F,
          dep.var.labels = c('VADER (sent.)', 'BERT (stance)', 'Ground Truth'),
          #column.labels = c('VADER (sent.)', 'BERT (stance)'),
          covariate.labels = c('Ideology (lib-cons)'))

sink('./../tables/Table_2.txt')
write(stargazer(m1, m2, m_ref, type = 'text',
          star.char = c('','',''),
          notes = '',
          digits = 2,
          notes.append = F,
          dep.var.labels = c('VADER (sent.)', 'BERT (stance)', 'Ground Truth'),
          covariate.labels = c('Ideology (lib-cons)')))
sink()



oos_ideology_score <- seq(from = -2.5, to = 2.5, length.out = 100)

pred_df1 <- as.data.frame(cbind(oos_ideology_score))
colnames(pred_df1) <- c('ideology_score')
preds <- predict(m1, newdata = pred_df1, type = 'response', se.fit = T)
pred_df1$predprob <- preds$fit
pred_df1$lwr <- preds$fit - 1.96*preds$se.fit
pred_df1$upr <- preds$fit + 1.96*preds$se.fit
pred_df1$model <- 'vader'

pred_df2 <- as.data.frame(cbind(oos_ideology_score))
colnames(pred_df2) <- c('ideology_score')
preds <- predict(m2, newdata = pred_df2, type = 'response', se.fit = T)
pred_df2$predprob <- preds$fit
pred_df2$lwr <- preds$fit - 1.96*preds$se.fit
pred_df2$upr <- preds$fit + 1.96*preds$se.fit
pred_df2$model <- 'bert'

pred_df_ref <- as.data.frame(cbind(oos_ideology_score))
colnames(pred_df_ref) <- c('ideology_score')
preds <- predict(m_ref, newdata = pred_df_ref, type = 'response', se.fit = T)
pred_df_ref$predprob <- preds$fit
pred_df_ref$lwr <- preds$fit - 1.96*preds$se.fit
pred_df_ref$upr <- preds$fit + 1.96*preds$se.fit
pred_df_ref$model <- 'ground truth'

plot_df <- rbind(pred_df1, pred_df2, pred_df_ref)

plot_df$trained_on <- c(rep('sentiment', 100), rep('stance', 100), rep('ground truth', 100))

# VADER sentiment v BERT stance
plot_df$model <- factor(plot_df$model, 
                        levels = c('ground truth', 'vader', 'bert'), 
                        labels = c('Human-coded Stance', 'VADER Sentiment Classifier', 'BERT Stance Classifier'))
ggplot(data = plot_df, aes(x = ideology_score, y = predprob, group = model))+
  geom_line(aes(color = model, linetype = model))+
  scale_color_manual(values = c('grey20', PANLightBlue, PANDarkBlue))+
  scale_fill_manual(values = c('grey50', PANLightBlue, PANDarkBlue))+
  scale_linetype_manual(values = c(1, 3, 5))+
  geom_ribbon(data=plot_df, aes(ymin = lwr, ymax = upr, fill = model), alpha = 0.15)+
  labs(y = 'Probability of Women\'s March Approval',
       x = NULL,
       color = NULL,
       linetype = NULL,
       fill = NULL)+ 
  scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(breaks = c(-2,-1,0,1,2), 
                     labels = c('Very Liberal', 'Liberal', 'Moderate', 'Conservative', 'Very Conservative'))+
  theme(legend.position="bottom")
ggsave('./../figures/F3.pdf', width = 8, height = 5)

###################################################################################################################
# MOOD OF THE NATION EXAMPLE

# produce counts for table 3
MOTN <- read.csv('./../data/MOTN_responses_groundtruth.csv', stringsAsFactors = F)

senti_stance_cor_table(MOTN$trump_stance_auto, MOTN$qpos, './../tables/Table_3.txt')

# function to produce label distribution tables in appendix (A5)
lab_dist_table <- function(DF, sent, stance, filename){
  t_df <- t(data.frame(
    as.numeric(table(DF[,sent], useNA = 'always')),
    as.numeric(table(DF[,stance], useNA = 'always')),
    as.numeric(table(DF$lexicoder_sentiment, useNA = 'always')),
    as.numeric(table(DF$vader_sentiment, useNA = 'always')),
    as.numeric(table(DF$SVM_sentiment, useNA = 'always')),
    as.numeric(table(DF$SVM_stance, useNA = 'always')),
    as.numeric(table(DF$BERT_sentiment, useNA = 'always')),
    as.numeric(table(DF$BERT_stance, useNA = 'always'))
  ))
  
  colnames(t_df) <- c('Negative', 'Positive', 'No Score')
  rownames(t_df) <- c('Ground Truth Sentiment', 'Ground Truth Stance',
                      'Lexicoder', 'VADER', 'SVM (sentiment-trained)',
                      'SVM (stance-trained)', 'BERT (sentiment-trained)',
                      'BERT (stance-trained)')
  
  t_df <- as.data.frame(t_df) %>% select('Negative', 'No Score', 'Positive')
  
  capture.output(stargazer(t_df, type = 'text', summary = F), file = filename)

  print(t_df)
}

# produce counts for table A5 (MOTN half)
lab_dist_table(MOTN, 'qpos', 'trump_stance_auto', './../tables/Table_A5_MOTN.txt')

# produce split counts for appendix (Table A8)
MOTN$moderate_sent <- ifelse(MOTN$vader_scores > 0.5, 0, 1)
MOTN$moderate_sent <- ifelse(MOTN$vader_scores < -0.5, 0, MOTN$moderate_sent)

# moderate
senti_stance_cor_table(MOTN$trump_stance_auto[MOTN$moderate_sent == 1], 
                       MOTN$qpos[MOTN$moderate_sent == 1],
                       './../tables/Table_A8_moderate.txt')

# extreme
senti_stance_cor_table(MOTN$trump_stance_auto[MOTN$moderate_sent == 0], 
                       MOTN$qpos[MOTN$moderate_sent == 0],
                       './../tables/Table_A8_extreme.txt')

# define function to compare classifiers; produce comparison tables
classifier_comparison_table <- function(filename, df, sentiment_var, stance_var, classifier_list, tiebreak_method_list='all_random', include_samplesize = FALSE){
  if(tiebreak_method_list=='all_random'){
    tiebreak_method_list <- rep('random', length(classifier_list))
  }
  
  sink(filename)
  for(i in 1:length(classifier_list)){
    if(include_samplesize == TRUE){
      if(tiebreak_method_list[i] == 'drop'){
        preds = df[,classifier_list[i]]
        N = length(preds[!is.na(preds)])
        samplesize_text = paste('Total Sample =', N, '\n', sep = ' ')
      }else{
        N = nrow(df)
        samplesize_text = paste('Total Sample =', N, '\n', sep = ' ')
      }
    }else{
      samplesize_text = ''
    }
    
    # check performance on sentiment task
    cat(paste(classifier_list[i], 'predicting sentiment,', tiebreak_method_list[i], 'tiebreak method:\n', sep = ' '))
    res <- report_results(df, sentiment_var, classifier_list[i], tiebreak_method_list[i])
    cat(samplesize_text)
    cat(paste('Average F1 Score: ', round(mean(res), 4), ' (se = ', round(se(res), 4), ')\n\n', sep = ''))
    
    # check performance on stance task
    cat(paste(classifier_list[i], 'predicting stance,', tiebreak_method_list[i], 'tiebreak method:\n', sep = ' '))
    res <- report_results(df, stance_var, classifier_list[i], tiebreak_method_list[i])
    cat(samplesize_text)
    cat(paste('Average F1 Score: ', round(mean(res), 4), ' (se = ', round(se(res), 4), ')\n\n', sep = ''))
    
  }
  sink()
  
  system(paste('cat ', filename))
}


# produce statistics for Table 4
set.seed(101)
classifier_comparison_table('./../tables/Table_4.txt',
                            MOTN,
                            'qpos',
                            'trump_stance_auto',
                            c('lexicoder_sentiment', 'vader_sentiment', 'SVM_sentiment',
                              'SVM_stance', 'BERT_sentiment', 'BERT_stance'))

# produce statistics for Table A6 (MOTN)
set.seed(101)
classifier_comparison_table('./../tables/Table_A6_MOTN.txt',
                            MOTN,
                            'qpos',
                            'trump_stance_auto',
                            c('lexicoder_sentiment', 'lexicoder_sentiment', 'lexicoder_sentiment',
                              'vader_sentiment', 'vader_sentiment', 'vader_sentiment',
                              'SVM_sentiment', 'SVM_stance', 'BERT_sentiment', 'BERT_stance'),
                            c('random', 'drop', 'strict',
                              'random', 'drop', 'strict',
                              'random', 'random', 'random', 'random'),
                            include_samplesize = TRUE)


# downstream regression; produce Table 5




MOTN$ideo5 <- recode(MOTN$ideo5,
                            'Very liberal' = 1,
                            'Liberal' = 2,
                            'Moderate' = 3,
                            'Not sure' = 3,
                            'Conservative' = 4,
                            'Very conservative' = 5)

m1 <- glm(random_fill_NA(lexicoder_sentiment) ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))
m2 <- glm(random_fill_NA(vader_sentiment) ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))
m3 <- glm(SVM_sentiment ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))
m4 <- glm(BERT_sentiment ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))
m5 <- glm(SVM_stance ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))
m6 <- glm(BERT_stance ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))
m_ref <- glm(trump_stance_auto ~ ideo5+as.factor(wavenum), data = MOTN, family = binomial(link = "logit"))



stargazer(m1, m2, m3, m4, m5, m6, m_ref, type = 'latex',
          star.char = c('','',''),
          notes = '',
          digits = 2,
          notes.append = F,
          omit = 'wavenum',
          omit.labels = 'Survey Wave FEs',
          #dep.var.labels = 'Trump Support',
          dep.var.labels = c('Lexicoder', 'VADER', 'SVM (sent.)', 'BERT (sent.)', 'SVM (stance)', 'BERT (stance)', 'Ground Truth Stance'),
          covariate.labels = c('Ideology (lib-cons)'))

sink('./../tables/Table_5.txt')
write(stargazer(m1, m2, m3, m4, m5, m6, m_ref, type = 'text',
          star.char = c('','',''),
          notes = '',
          digits = 2,
          notes.append = F,
          omit = 'wavenum',
          omit.labels = 'Survey Wave FEs',
          #dep.var.labels = 'Trump Support',
          dep.var.labels = c('Lexicoder', 'VADER', 'SVM (sent.)', 'BERT (sent.)', 'SVM (stance)', 'BERT (stance)', 'Ground Truth Stance'),
          covariate.labels = c('Ideology (lib-cons)')))
sink()

# produce Figure 4
oos_ideo5 <- seq(from = 0.5, to = 5.5, length.out = 100)
oos_wavenum <- rep(12, 100)
pred_df4 <- as.data.frame(cbind(oos_ideo5, oos_wavenum))
colnames(pred_df4) <- c('ideo5', 'wavenum')
preds <- predict(m4, newdata= pred_df4, type = 'response', se.fit = T)
pred_df4$predprob <- preds$fit
pred_df4$lwr <- preds$fit - 1.96*preds$se.fit
pred_df4$upr <- preds$fit + 1.96*preds$se.fit
pred_df4$model <- 'BERT'
pred_df4$trained_on <- 'BERT Sentiment Classifier'

pred_df6 <- as.data.frame(cbind(oos_ideo5, oos_wavenum))
colnames(pred_df6) <- c('ideo5', 'wavenum')
preds <- predict(m6, newdata= pred_df6, type = 'response', se.fit = T)
pred_df6$predprob <- preds$fit
pred_df6$lwr <- preds$fit - 1.96*preds$se.fit
pred_df6$upr <- preds$fit + 1.96*preds$se.fit
pred_df6$model <- 'BERT'
pred_df6$trained_on <- 'BERT Stance Classifier'

pred_df_ref <- as.data.frame(cbind(oos_ideo5, oos_wavenum))
colnames(pred_df_ref) <- c('ideo5', 'wavenum')
preds <- predict(m_ref, newdata= pred_df_ref, type = 'response', se.fit = T)
pred_df_ref$predprob <- preds$fit
pred_df_ref$lwr <- preds$fit - 1.96*preds$se.fit
pred_df_ref$upr <- preds$fit + 1.96*preds$se.fit
pred_df_ref$model <- 'Self-reported Stance'
pred_df_ref$trained_on <- 'Self-reported Stance'

pred_df <- rbind(pred_df_ref, pred_df4, pred_df6)
pred_df$trained_on <- factor(pred_df$trained_on, levels = c('Self-reported Stance', 'BERT Sentiment Classifier', 'BERT Stance Classifier'))

ggplot(data = pred_df, aes(x = ideo5, y = predprob, group = trained_on))+
  geom_line(aes(color = trained_on, linetype = trained_on))+
  scale_color_manual(values = c('grey20', PANLightBlue, PANDarkBlue))+
  scale_fill_manual(values = c('grey50', PANLightBlue, PANDarkBlue))+
  scale_linetype_manual(values = c(1, 3, 5))+
  geom_ribbon(data=pred_df, aes(ymin = lwr, ymax = upr, fill = trained_on), alpha = 0.15)+
  labs(y = 'Probability of Trump Approval',
       x = NULL,
       color = NULL,
       linetype = NULL,
       fill = NULL)+ 
  scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(breaks = c(1,2,3,4,5), 
                     labels = c('Very Liberal', 'Liberal', 'Moderate', 'Conservative', 'Very Conservative'))+
  theme(legend.position="bottom")

ggsave('./../figures/F4.pdf', width = 8, height=5)

####################################################################################################################
# Kavanaugh Tweets Example

# produce counts for table 6
KAV <- read.csv('./../data/kavanaugh_tweets_groundtruth.csv', stringsAsFactors = F)
senti_stance_cor_table(KAV$stance, KAV$sentiment, './../tables/Table_6.txt')


# produce counts for table A5 (KAV half)
lab_dist_table(KAV, 'sentiment', 'stance', './../tables/Table_A5_KAV.txt')


# produce split counts for appendix (Table A9)
KAV$moderate_sent <- ifelse(KAV$vader_scores > 0.5, 0, 1)
KAV$moderate_sent <- ifelse(KAV$vader_scores < -0.5, 0, KAV$moderate_sent)


# moderate
senti_stance_cor_table(KAV$stance[KAV$moderate_sent == 1], 
                       KAV$sentiment[KAV$moderate_sent == 1],
                       './../tables/Table_A9_moderate.txt')

# extreme
senti_stance_cor_table(KAV$stance[KAV$moderate_sent == 0], 
                       KAV$sentiment[KAV$moderate_sent == 0],
                       './../tables/Table_A9_extreme.txt')


# produce classifier comparison statistics for Table 7
set.seed(101)
classifier_comparison_table('./../tables/Table_7.txt',
                            KAV, 
                            'sentiment',
                            'stance',
                            c('lexicoder_sentiment', 'vader_sentiment', 'SVM_sentiment',
                              'SVM_stance', 'BERT_sentiment', 'BERT_stance'))

# produce classifier comparison statistics for Table A6 (KAV)
set.seed(101)
classifier_comparison_table('./../tables/Table_A6_KAV.txt',
                            KAV,
                            'sentiment',
                            'stance',
                            c('lexicoder_sentiment', 'lexicoder_sentiment', 'lexicoder_sentiment',
                              'vader_sentiment', 'vader_sentiment', 'vader_sentiment',
                              'SVM_sentiment', 'SVM_stance', 'BERT_sentiment', 'BERT_stance'),
                            c('random', 'drop', 'strict',
                              'random', 'drop', 'strict',
                              'random', 'random', 'random', 'random'),
                            include_samplesize = TRUE)



# downstream regression; produce Table 8
analysis_df <- read.csv('./../data/kavanaugh_tweets_analysis_tweetscores.csv', stringsAsFactors = F)
analysis_df$ideology_score <- ifelse(analysis_df$ideology_score == Inf, 2.5, analysis_df$ideology_score)
analysis_df$ideology_score <- ifelse(analysis_df$ideology_score == '#NAME?', -2.5, analysis_df$ideology_score)
analysis_df$ideology_score <- ifelse(analysis_df$ideology_score == -Inf, -2.5, analysis_df$ideology_score)
analysis_df$ideology_score <- as.numeric(analysis_df$ideology_score)

m1 <- glm(data = analysis_df, random_fill_NA(lexicoder_sentiment)~ideology_score, family = binomial(link = 'logit'))
m2 <- glm(data = analysis_df, random_fill_NA(vader_sentiment)~ideology_score, family = binomial(link = 'logit'))
m3 <- glm(data = analysis_df, SVM_sentiment~ideology_score, family = binomial(link = 'logit'))
m4 <- glm(data = analysis_df, BERT_sentiment~ideology_score, family = binomial(link = 'logit'))
m5 <- glm(data = analysis_df, SVM_stance~ideology_score, family = binomial(link = 'logit'))
m6 <- glm(data = analysis_df, BERT_stance~ideology_score, family = binomial(link = 'logit'))
m_ref <- glm(data = analysis_df, stance~ideology_score, family = binomial(link = 'logit'))

stargazer(m1, m2, m3, m4, m5, m6, m_ref, type = 'latex',
          star.char = c('','',''),
          notes = '',
          notes.append = F,
          digits = 2,
          dep.var.labels = c('Lexicoder', 'VADER', 'SVM (sent.)', 'BERT (sent.)', 'SVM (stance)', 'BERT (stance)', 'Ground Truth'),
          #column.labels = c('Lexicoder', 'VADER', 'SVM (sent.)', 'BERT (sent.)', 'SVM (stance)', 'BERT (stance)', 'Ground Truth'),
          covariate.labels = c('Ideology'))

sink('./../tables/Table_8.txt')
write(stargazer(m1, m2, m3, m4, m5, m6, m_ref, type = 'text',
          star.char = c('','',''),
          notes = '',
          notes.append = F,
          digits = 2,
          dep.var.labels = c('Lexicoder', 'VADER', 'SVM (sent.)', 'BERT (sent.)', 'SVM (stance)', 'BERT (stance)', 'Ground Truth'),
          #column.labels = c('Lexicoder', 'VADER', 'SVM (sent.)', 'BERT (sent.)', 'SVM (stance)', 'BERT (stance)', 'Ground Truth'),
          covariate.labels = c('Ideology')))
sink()

# produce Figure 5
oos_ideology_score <- seq(from = -2.5, to = 2.5, length.out = 100)

pred_df4 <- as.data.frame(cbind(oos_ideology_score))
colnames(pred_df4) <- c('ideology_score')
preds <- predict(m4, newdata = pred_df4, type = 'response', se.fit = T)
pred_df4$predprob <- preds$fit
pred_df4$lwr <- preds$fit - 1.96*preds$se.fit
pred_df4$upr <- preds$fit + 1.96*preds$se.fit
pred_df4$model <- 'BERT'

pred_df6 <- as.data.frame(cbind(oos_ideology_score))
colnames(pred_df6) <- c('ideology_score')
preds <- predict(m6, newdata = pred_df6, type = 'response', se.fit = T)
pred_df6$predprob <- preds$fit
pred_df6$lwr <- preds$fit - 1.96*preds$se.fit
pred_df6$upr <- preds$fit + 1.96*preds$se.fit
pred_df6$model <- 'BERT'

pred_df_ref <- as.data.frame(cbind(oos_ideology_score))
colnames(pred_df_ref) <- c('ideology_score')
preds <- predict(m_ref, newdata= pred_df_ref, type = 'response', se.fit = T)
pred_df_ref$predprob <- preds$fit
pred_df_ref$lwr <- preds$fit - 1.96*preds$se.fit
pred_df_ref$upr <- preds$fit + 1.96*preds$se.fit
pred_df_ref$model <- 'Ground Truth'

pred_df <- rbind(pred_df4, pred_df6, pred_df_ref)

pred_df$trained_on <- c(rep('BERT Sentiment Classifier', 100), rep('BERT Stance Classifier', 100), rep('Ground Truth Stance', 100))
pred_df$trained_on <- factor(pred_df$trained_on, 
                        levels = c('Ground Truth Stance', 'BERT Sentiment Classifier', 'BERT Stance Classifier'),
                        labels = c('Self-reported Stance', 'BERT Sentiment Classifier', 'BERT Stance Classifier'))

# BERT sentiment v stance
ggplot(data = pred_df, aes(x = ideology_score, y = predprob, group = trained_on))+
  geom_line(aes(color = trained_on, linetype = trained_on))+
  scale_color_manual(values = c('grey20', PANLightBlue, PANDarkBlue))+
  scale_fill_manual(values = c('grey50', PANLightBlue, PANDarkBlue))+
  scale_linetype_manual(values = c(1, 3, 5))+
  geom_ribbon(data=pred_df, aes(ymin = lwr, ymax = upr, fill = trained_on), alpha = 0.15)+
  labs(y = 'Probability of Kavanaugh Approval',
       x = NULL,
       color = NULL,
       linetype = NULL,
       fill = NULL)+ 
  scale_y_continuous(limits=c(0,1))+
  scale_x_continuous(breaks = c(-2,-1,0,1,2), 
                     labels = c('Very Liberal', 'Liberal', 'Moderate', 'Conservative', 'Very Conservative'))+
  theme(legend.position="bottom")

ggsave('./../figures/F5.pdf', width = 8, height=5)


