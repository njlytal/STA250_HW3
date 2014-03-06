# STA 250 HW 3: CLASSIFICATION METHODS

# ********* INITIAL DATA PREP ***********
# Import the training sample
setwd("~/Google Drive/STA 250/HW3") # location of file (varies)
train.sample <- read.csv("train-sample.csv")

# Important libraries for tests
library(adabag)
library(randomForest)
library(rpart)
library(qdap)

# Convert status into either OPEN or CLOSED
status = as.numeric(train.sample$OpenStatus)
for(i in 1:length(status))
{
    if(status[i]==4){
        status[i] = 1 # OPEN
    }else{
        status[i] = 0 # CLOSED
    }
}
train.sample$OpenStatus = status

# Identifies all words used in the titles (qdap)
titles = train.sample$Title
aw.title = all_words(titles)

# Sorts all word counts to find the most common ones
head(aw.title[with(aw.title,order(-FREQ)),],32)

# Top Recurring Non-Stop Words: using, c, php, what,
# java, file, android, jquery, code, data
# Use to compose a wordlist
wordlist = c('c','php','what','java','file','android',
             'jquery','code','data', 'get')



# This code scans every title for the presence of the words
# in wordlist, returning a TRUE or FALSE result for each.

# Ideally applied to all titles, but too computationally
# intense to complete -- Computer freezes after ~50 min.
word.tf = lapply(titles[1:10000],
                 FUN=function(x) wordlist %in% all_words(x)$WORD)

# Turns the word results into a data frame
word.df = as.data.frame(do.call(rbind,word.tf))
names(word.df)=wordlist

# Combines with original data set & simplifies for use
# in randomForest
x = cbind(train.sample[1:10000,],word.df)
train.mod = x[ , -which(names(x) %in% c("Title","BodyMarkdown"))]
train.mod = train.mod[,c(1:6,12:23)]
train.mod = train.mod[,-7]



# ************* CLASSIFICATION METHODS **************

# Convert to a factor to use classification
train.mod$OpenStatus = factor(train.mod$OpenStatus)


# Arguments specified rather than doing OpenStatus~.
# to avoid ambiguity

# Random Forests
rf.data = randomForest(OpenStatus~ReputationAtPostCreation +
            OwnerUndeletedAnswerCountAtPostTime +
            OwnerUserId + c + php + what + java +
            file + android + jquery + code + data + get,
            data=train.mod, ntree=100, do.trace=10)

# NOTE: A barplot is not ideal, but it does easily allow
# variable names to fit and be compared
par(mar = c(4,14,4,2)) # Positioning
barplot(sort(rf.data$importance, decreasing = TRUE),
        main = "Mean Gini Decrease (Random Forests)",
        horiz = TRUE, las = 1,
        names.arg=(names(train.mod[,-c(1,2,4,7)])))

# Boosting
boost.data = boosting(OpenStatus~ReputationAtPostCreation +
         OwnerUndeletedAnswerCountAtPostTime +
         OwnerUserId + c + php + what + java +
         file + android + jquery + code + data + get,
         data=train.mod,
         control=rpart.control(maxdepth=5,cp=0.001),
         mfinal = 30)

# Importance of variables according to boosting algorithm
barplot(sort(boost.data$importance, decreasing = TRUE),
        main = "Relative Importance of Variables (Boosting)",
        horiz = TRUE, las = 1, xlim = c(0, 65))

test = rpart(OpenStatus~ReputationAtPostCreation +
                   OwnerUndeletedAnswerCountAtPostTime +
                   OwnerUserId + c + php + what + java +
                   file + android + jquery + code + data + get,
               data=train.mod, method='class',
               control=rpart.control(maxdepth=5,cp=0.001))
par(xpd=NA)
plot(test)
text(test, use.n=TRUE)

# ********** FORMER NLP ATTEMPTS ***********

# Always use before rJava to increase memory accessible
options(java.parameters = "-Xmx4g")

titles = train.sample$Title
str = as.String(titles)
sent_ann = Maxent_Sent_Token_Annotator()
sents = annotate(str,sent_ann)
word_ann = Maxent_Word_Token_Annotator()

# The word annotator failed to produce results
# after nearly 30 minutes, and caused R to hang
# as well
words = annotate(str, word_ann, sents)

# From here we could proceed to parts of speech
# annotation, and would compare the presence of
# each word in both the title and the Body Markdown
# rather than just a select few