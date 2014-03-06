# STA 250 HW 3: CLASSIFICATION METHODS

# NOTES
# Focus on Reputation, actual post content, and post status
# (which includes why it was closed if applicable)

# METHODS
# Boosting & Random Forests?

# Using NLP

# EX: x = String(train.sample$Title[1])
# x now outputs: "what is the best way to connect my
# application with kernel?"

# Experimenting with rpart
# define priors, work with rpart.control to inhibit pruning, etc.
# alter/try out method variants

# Needed to allow Java to access more memory
options( java.parameters = "-Xmx4g" ) 

setwd("~/Google Drive/STA 250/HW3")
train.sample <- read.csv("train-sample.csv",
                  colClasses=c("numeric", "character", "numeric",
                               "character", "numeric", "numeric",
                               rep("character",9 )))


status = as.numeric(train.sample$OpenStatus)
# Convert status into either OPEN or CLOSED
for(i in 1:length(status))
{
    if(status[i]==4){
        status[i] = 1
    }else{
        status[i] = 0
    }
}
train.sample$OpenStatus = status

# Bag O Words test
# test.t = train.sample$BodyMarkdown[1:1000]
#words.bm = bag_o_words(test.t)



titles = train.sample$Title
words.title = bag_o_words(titles) # Takes about 1 minute

aw.title = all_words(titles)

# Sorts all word counts to find the most common ones
head(aw.title[with(aw.title,order(-FREQ)),],32)

# Top Recurring Words: using, c, php, what, java, file,
# android, jquery, code, data
wordlist = c('c','php','what','java','file','android',
             'jquery','code','data', 'get')

start = proc.time()
# This code scans every title for the presence of the words
# in wordlist, returning a TRUE or FALSE result for each.
word.tf = lapply(titles, FUN=function(x) wordlist %in% all_words(x)$WORD)
time = proc.time() - start

word.df = word.df = as.data.frame(do.call(rbind,word.tf))
names(word.df)=wordlist
new.train = cbind(train.sample,word.df)


# ***** START TEST AREA *****
start = proc.time()
word.tf = lapply(titles[1:10000], FUN=function(x) wordlist %in% all_words(x)$WORD)
time = proc.time() - start

# Turns the word results into a data frame
word.df = as.data.frame(do.call(rbind,word.tf))
names(word.df)=wordlist

x = cbind(train.sample[1:10000,],word.df)
train.mod = x[ , -which(names(x) %in% c("Title","BodyMarkdown"))]

(wordlist %in% all_words(titles)$WORD)

train.mod = train.mod[,c(1:6,12:23)]

randomForest(OpenStatus~.,data=train.mod, ntree=3)

test = rpart(OpenStatus ~ ., data = train.mod, method = "anova",
             minsplit = 5)

test = rpart(OpenStatus ~ ReputationAtPostCreation +
                 OwnerUndeletedAnswerCountAtPostTime +
                 OwnerUserId + c + php + what + java +
                 file + android + jquery + code + data + get,
             data = train.mod,
             method = "anova", minsplit=10)
# ***** END TEST AREA *****


test = rpart(OpenStatus ~ ReputationAtPostCreation +
                 OwnerUndeletedAnswerCountAtPostTime +
                 OwnerUserId, data = train.sample,
             method = "anova", minsplit=5)
par(xpd=NA)
plot(test)
text(test, use.n=TRUE, all=TRUE)

test = randomForest(OpenStatus ~ ReputationAtPostCreation +
                    OwnerUndeletedAnswerCountAtPostTime +
                    OwnerUserId, data = train.sample,
                    ntree = 30)


# Always use before rJava
options(java.parameters = "-Xmx4g")

titles = train.sample$Title
str = as.String(titles)
sent_ann = Maxent_Sent_Token_Annotator()
sents = annotate(str,sent_ann)
word_ann = Maxent_Word_Token_Annotator()

words = annotate(str, word_ann, sents)

pos_ann = Maxent_POS_Tag_Annotator()
pos = annotate(str, pos_ann, words) 
i = pos$type == 'word'
ww = str[pos][i]
cbind(ww, unlist(pos$features[i]))

?Maxent_Word_Token_Annotator
