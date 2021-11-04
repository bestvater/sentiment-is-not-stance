####################################################################################################################
# Bestvater & Monroe; Sentiment != Stance
# 3_tweetscores.R
# R script to implement Tweetscores ideology scaling for Twitter users (see Barbera 2015)
####################################################################################################################

# SETUP
library(tweetscores)

consumer_key <- ""
consumer_secret <-""
access_token <- ""
access_secret <- "" 

my_oauth <- list(consumer_key = consumer_key,
                 consumer_secret = consumer_secret,
                 access_token= access_token,
                 access_token_secret = access_secret)

tweetscore <- function(user, oauth){
  friends <- getFriends(user, oauth = oauth)
  score <- estimateIdeology2(user, friends)
  return(score)
}

####################################################################################################################
# WOMEN'S MARCH REPLICATION/EXTENSION

analysis <- read.csv('./../data/WM_tweets_analysis_tweetscores.csv')

users <- analysis$user_screen_name

ideology_score <- c()

for(i in 1:length(users)){
  user <- users[i]
  check <- try(tweetscore(user, my_oauth))
  if(class(check) == 'try-error'){
    ideology_score <- c(ideology_score, NA)
  }else{
    ideology_score <- c(ideology_score, check)
  }
}

analysis$ideology_score <- ideology_score

write.csv(analysis, './../data/WM_tweets_analysis_tweetscores.csv', row.names = FALSE)


####################################################################################################################
# Kavanaugh Tweets Example

analysis <- read.csv('./../data/kavanaugh_tweets_analysis_tweetscores.csv')

users <- analysis$user_screen_name

ideology_score <- c()

for(i in 1:length(users)){
  user <- users[i]
  check <- try(tweetscore(user, my_oauth))
  if(class(check) == 'try-error'){
    ideology_score <- c(ideology_score, NA)
  }else{
    ideology_score <- c(ideology_score, check)
  }
}

analysis$ideology_score <- ideology_score

write.csv(analysis, './../data/kavanaugh_tweets_analysis_tweetscores.csv', row.names = FALSE)
