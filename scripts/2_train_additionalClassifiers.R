####################################################################################################################
# Bestvater & Monroe; Sentiment != Stance
# 2_additionalClassifiers.r
# R script to run Lexicoder out of the Quanteda library
####################################################################################################################

# SETUP

require(quanteda)

lexicode <- function(text_vector){
  toks <- tokens(corpus(as.character(text_vector)), remove_punct = TRUE)
  toks_lsd <- tokens_lookup(toks, dictionary =  data_dictionary_LSD2015[1:2])
  dfmat_lsd <- dfm(toks_lsd)
  lsd_df = convert(dfmat_lsd, to = 'data.frame')
  lsd_df$lexicoder_sentiment <- ifelse(lsd_df$positive < lsd_df$negative, 0, NA)
  lsd_df$lexicoder_sentiment <- ifelse(lsd_df$positive > lsd_df$negative, 1, lsd_df$lexicoder_sentiment)
  
  return(lsd_df$lexicoder_sentiment)
}

####################################################################################################################
# MOOD OF THE NATION EXAMPLE

MOTN <- read.csv('./../data/MOTN_responses_groundtruth.csv')
MOTN$lexicoder_sentiment <- lexicode(MOTN$edits_clean_text)

write.csv(MOTN, './../data/MOTN_responses_groundtruth.csv', row.names = FALSE)


####################################################################################################################
# Kavanaugh Tweets Example

KAV <- read.csv('./../data/kavanaugh_tweets_groundtruth.csv')
KAV$lexicoder_sentiment <- lexicode(KAV$text)

write.csv(KAV, './../data/kavanaugh_tweets_groundtruth.csv', row.names = FALSE)

analysis <- read.csv('./../data/kavanaugh_tweets_analysis_tweetscores.csv')
analysis$lexicoder_sentiment <- lexicode(analysis$text)

write.csv(analysis, './../data/kavanaugh_tweets_analysis_tweetscores.csv', row.names = FALSE)
