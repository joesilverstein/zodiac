# Zodiac Code Sample
# Joe Silverstein
# 5-23-16

library(randomForest)
library(e1071)
library(vcd)
library(tm)
library(SnowballC)
# library(adabag)
library(neuralnet)
library(gbm)

setwd("/Users/joesilverstein/Google Drive/Jobs")

# When giving final version, first write code to remove all commas and %'s
# (since he's using a different dataset)

# Also need to make sure that all 31 bag of words count variables are included in his dataset, so he can do prediction.

# First, convert all fields in 'anonymized data.xlsx' to 'General' in Excel and resave.
# Then run this code.
df = read.csv("anonymized data.csv", na.strings = c("", "n/a", "#VALUE!")) 
# I checked, and the '#VALUE!' was supposed to be missing, but got converted weirdly when converted to CSV.

# Convert "paid_in_full*" to "paid_in_full"
for (i in 1:nrow(df)) {
  if (df$Payment.Status[i] == "paid_in_full*") df$Payment.Status[i] = "paid_in_full"
}
df$Payment.Status = droplevels(df$Payment.Status)

classes = lapply(df, class)

# Frequencies and distribution of payment statuses
dfPaymentStatusTable = data.frame(table(df$Payment.Status))
dfPaymentStatusTable$Perc = dfPaymentStatusTable$Freq / nrow(df)
dfPaymentStatusTable

# Only keep columns with with less than 10% missing
dfFewerMissing = df[ lapply(df, function(x) sum(is.na(x)) / length(x) ) < 0.1 ]

# Only keep complete cases
dfComplete = dfFewerMissing[complete.cases(dfFewerMissing), ]

# See how many text memos for Use.Of.Funds there are
length(levels(dfComplete$Use.of.Funds))

## May want to convert Use.of.Funds to features that are useable in regression using LSA
# http://www-stat.wharton.upenn.edu/~stine/research/regressor.pdf
# (The above link also explains why LSA works. Come back to this if it works.)

# prepare corpus
corpus = Corpus(VectorSource(dfComplete$Use.of.Funds))
# The latest version of tm (0.60) made it so you can't use functions with tm_map that operate on simple character values any more.
# http://stackoverflow.com/questions/24771165/r-project-no-applicable-method-for-meta-applied-to-an-object-of-class-charact
corpus = tm_map(corpus, content_transformer(tolower)) # convert to lowercase
corpus = tm_map(corpus, removePunctuation) # remove punctuation
# Remove "stopwords" from text
# Stopwords are words to remove from the text. They are given here:
# http://jmlr.csail.mit.edu/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
corpus = tm_map(corpus, function(x) removeWords(x, stopwords("english")))
# Stem words using Porter's stemming algorithm
corpus = tm_map(corpus, stemDocument, language = "english")

# Create Bag Of Words matrix
bagOfWordsMat = t(as.matrix(TermDocumentMatrix(corpus)))
# Only 31 relevant word stems are used! Might not even need to reduce the dimensionality.

# Don't convert the bag of words variables to factors, because the bag of word variables are not categorical.

# Add word type counts to dataset
dfComplete = cbind(dfComplete, bagOfWordsMat)

# Remove Use.of.Funds
dfComplete$Use.of.Funds = NULL

# Remove X.ID, since it's equivalent to a row identifier
dfComplete$X.ID = NULL

# Find how many levels each factor has
sapply(dfComplete[,sapply(dfComplete, is.factor)], nlevels)

# See how many are of each Payment.Status
(paymentStatusTable = data.frame(table(dfComplete$Payment.Status)))

(percentCurrent = paymentStatusTable[1, 2] / nrow(dfComplete))

# Frequencies and distribution of payment statuses
dfCompletePaymentStatusTable = data.frame(table(dfComplete$Payment.Status))
dfCompletePaymentStatusTable$Perc = dfCompletePaymentStatusTable$Freq / nrow(dfComplete)
dfCompletePaymentStatusTable
# Looks pretty similar to dfPaymentStatusTable, so the missing data appears to be missing at random.

# Misclassification error of predicting all Current.
# Obviously, model needs to be able to beat this.
(baselineMisclassification = 1 - percentCurrent)

## List features by their level of association with Payment.Status

# For continuous variables, use the R^2 of a linear regression as a measure of association (because there is no such thing as "correlation" in this case)
# http://stats.stackexchange.com/questions/119835/correlation-between-a-nominal-iv-and-a-continuous-dv-variable/124618#124618

# Throw out factors, since I'm using Cramer's V for those
isFactor = sapply(dfComplete, is.factor)

RdfComplete = dfComplete[, !isFactor]
RdfComplete$Payment.Status = dfComplete$Payment.Status

tmpdf = data.frame(matrix(-1, ncol(RdfComplete), 1))
Rdf = data.frame(cbind(names(RdfComplete), tmpdf)) # initialization
names(Rdf) = c("Variable", "R")
dfNames = names(RdfComplete)
for (i in 1:ncol(RdfComplete)) {
  name = dfNames[i]
  formula = paste(name, " ~ Payment.Status")
  reg = lm(formula, data = RdfComplete)
  Rdf[i, 2] = sqrt(summary(reg)$r.squared) # correlation between the observed durations, and the ones predicted (fitted) by our model
}
Rdf = Rdf[order(-Rdf$R), ]

# For categorical variables, use Cramer's V

cramerdfComplete = dfComplete[, isFactor]
# Move Payment.Status to the last column so it doesn't throw an error when trying to compute Cramer's V with itself
# (only necessary if adding more factors)
Payment.Status = cramerdfComplete[, 4]
cramerdfComplete = cramerdfComplete[, -4]
cramerdfComplete = cbind(cramerdfComplete, Payment.Status)
tmpdf = data.frame(matrix(-1, ncol(cramerdfComplete), 1))
cramerdf = data.frame(cbind(names(cramerdfComplete), tmpdf)) # initialization
names(cramerdf) = c("Variable", "CramersV")
dfNames = names(cramerdfComplete)
for (i in 1:(ncol(cramerdfComplete)-1)) {
  name = dfNames[i]
  formula = paste("~Payment.Status +", name)
  tab = xtabs(formula, data = cramerdfComplete) # assocstats function takes contingency table as argument
  cramerdf[i, 2] = summary(assocstats(tab))$object$cramer
}
cramerdf = cramerdf[1:(nrow(cramerdf)-1), ]
cramerdf = cramerdf[order(-cramerdf$CramersV), ]

# Note that Cramer's V is very low for the association between Payment.Status and Credit.rating, 
# indicating that the credit rating might not be so good. Should first estimate the final payment
# status before making this conclusion, though.
# State and industry are much more associated with the payment status than the credit rating.

## Impute the Missing Values (work on this later)

# http://www.r-bloggers.com/imputing-missing-data-with-r-mice-package/
# See code in Documents folder

### Part 1: Prediction

## Train model

## divide into training and test data (80-20 split)
n = nrow(dfComplete)
indextrain = sample(1:n, round(0.8*n), replace=FALSE)
train = dfComplete[ indextrain, ]
test = dfComplete[-indextrain, ]
trainX = dfComplete[ indextrain, ]
testX = dfComplete[-indextrain, ]
trainX$Payment.Status = NULL
testX$Payment.Status = NULL
trainY = dfComplete[ indextrain, "Payment.Status"]
testY= dfComplete[-indextrain, "Payment.Status"]

# Random Forest

set.seed(42)
rfModel = randomForest(Payment.Status ~ ., data = dfComplete, importance = TRUE, proximity = TRUE)

# Rank variables by importance:
rfImportance = importance(rfModel)

predict.outofbag = predict(rfModel)

table(predict.outofbag)

plot(predict.outofbag)
# Worse than baseline misclassification error. There should be something better.

# Out of bag misclassification rate:
(rfOutOfBagMisclassification = 1 - mean(predict.outofbag == dfComplete$Payment.Status))

# Out of bag is usually an overestimate of the true misclassification rate, so also try on test dataset
rfModelCV = randomForest(Payment.Status ~ ., data = train)
predictTest = predict(rfModelCV, newdata = test)
(rfTestMisclassification = 1 - mean(predictTest == dfComplete$Payment.Status))
# Same as before.

# Try removing the variables with low values of R and train the random forest again
summary(Rdf$R)
keepInRF = as.character(droplevels(subset(Rdf, R > 0.045)$Variable))
dfSmall = dfComplete[, keepInRF]
dfSmall$Payment.Status = dfComplete$Payment.Status
dfSmall$State = dfComplete$State
dfSmall$Industry = dfComplete$Industry
dfSmall$Credit.rating = dfComplete$Credit.rating
rfSmall = randomForest(Payment.Status ~ ., data = dfSmall)
predictSmallOOB = predict(rfSmall)
table(predictSmallOOB)
(rfSmallMisclassification = 1 - mean(predictSmallOOB == dfComplete$Payment.Status))

# Create training and test versions of dfSmall
trainSmall = dfSmall[ indextrain, ]
testSmall = dfSmall[-indextrain, ]

# SVM

svmModel = svm(Payment.Status ~ ., data = train)
svmPred = predict(svmModel, newdata = test)

# Misclassification rate on test dataset
(svmTestMisclassification = 1 - mean(svmPred == test$Payment.Status)) # Worse than baseline

# Small version
svmSmall = svm(Payment.Status ~ ., data = trainSmall)
svmPredictSmall = predict(svmSmall, newdata = testSmall)
table(svmPredictSmall)
(svmSmallMisclassification = 1 - mean(svmPredictSmall == testSmall$Payment.Status))

# Naive Bayes
# (Be careful because it assumes independence between the features -- will maybe need to do a lot of preprocessing of inputs using ICA or something like it)

naiveBayesModel = naiveBayes(Payment.Status ~ ., data = trainSmall)
naiveBayesPred = predict(naiveBayesModel, newdata = testSmall)
table(naiveBayesPred)
(naiveBayesMisclassification = 1 - mean(naiveBayesPred == testSmall$Payment.Status))
# Horrible. Almost everything is classified incorrectly.

# AdaBoost

# Runs too slowly to actually use this.
# adaBoostModel = boosting(Payment.Status ~ ., data = trainSmall)
# adaBoostPred = predict(adaBoostModel, newdata = testSmall)
# table(adaBoostPred)
# (adaBoostMisclassification = 1 - mean(adaBoostPred == testSmall$Payment.Status))

# Gradient Boosted Trees
# This implements the generalized boosted modeling framework. Boosting is the process
# of iteratively adding basis functions in a greedy fashion so that each additional basis function further
# reduces the selected loss function. This implementation closely follows Friedman’s Gradient
# Boosting Machine (Friedman, 2001).

gbtModel = gbm(Payment.Status ~ ., data = trainSmall, distribution = "multinomial", n.trees = 500)
gbtPred = predict(gbtModel, newdata = testSmall, n.trees = 500)
p.gbtPred = apply(gbtPred, 1, which.max)
table(p.gbtPred)
# They're all predicted to be "current"

# Remember to output confusion matrix

### Part 2: Model Evaluation

# Can't use Tobit to get final payment status (defaulted, paid_in_full_late, paid_in_full_on_time), where a value of "current" indicates censoring
# because it isn't a censored (or truncated) regression problem (see notes)

# There is no way to predict the final payment status for Payment.Status \in {"current", "late"}, but we do know the final payment status of obs with
# Payment.Status \in {"defaulted", "paid_in_full"}. So restrict to that data and use Cramer's V to see how strong the association between
# Credit.rating and Payment.Status is for that subset of observations. Since they are opposites, the credit rating should be able to distinguish the two.
dfFinalPaymentStatus = droplevels(subset(dfComplete, Payment.Status == "defaulted" | Payment.Status == "paid_in_full"))
tab = xtabs(~ Payment.Status + Credit.rating, data = dfFinalPaymentStatus) # assocstats function takes contingency table as argument
tab # actually looks okay, especially in the A and A+ range
(cramerPaymentStatusCreditRating = summary(assocstats(tab))$object$cramer)
# Actually the credit rating isn't so bad at the high end. It distinguishes defaulted and paid in full decently.

# Now do same analysis with R and Cramer's V as before, but on subset
# Note that we can't directly compare Cramer's V with R, because they are different measures. So only use the categorical variables.

# Throw out factors, since I'm using Cramer's V for those
isFactor = sapply(dfFinalPaymentStatus, is.factor)
cramerdfFinalPaymentStatus = dfFinalPaymentStatus[, isFactor]
# Move Payment.Status to the last column so it doesn't throw an error when trying to compute Cramer's V with itself
# (only necessary if adding more factors)
Payment.Status = cramerdfFinalPaymentStatus[, 4]
cramerdfFinalPaymentStatus = cramerdfFinalPaymentStatus[, -4]
cramerdfFinalPaymentStatus = cbind(cramerdfFinalPaymentStatus, Payment.Status)
tmpdf = data.frame(matrix(-1, ncol(cramerdfFinalPaymentStatus), 1))
cramerdf = data.frame(cbind(names(cramerdfFinalPaymentStatus), tmpdf)) # initialization
names(cramerdf) = c("Variable", "CramersV")
dfNames = names(cramerdfFinalPaymentStatus)
for (i in 1:(ncol(cramerdfFinalPaymentStatus)-1)) {
  name = dfNames[i]
  formula = paste("~Payment.Status +", name)
  tab = xtabs(formula, data = cramerdfFinalPaymentStatus) # assocstats function takes contingency table as argument
  cramerdf[i, 2] = summary(assocstats(tab))$object$cramer
}
cramerdf = cramerdf[1:(nrow(cramerdf)-1), ]
cramerdf = cramerdf[order(-cramerdf$CramersV), ]

# State and Industry are much more associated with Payment Status than is credit rating. Look at crosstabs:
xtabs(~ Payment.Status + State, data = cramerdfFinalPaymentStatus) 
# Highly likely to pay in full (early!): CA, GA, IL, MI, NY, TX
# Likely to default (early!): maybe NV, but there isn't really enough data on defaults
xtabs(~ Payment.Status + State + Credit.rating, data = cramerdfFinalPaymentStatus) 
# Credit rating of A or A+ predicts the final payment status well. But credit ratings of B, C, or D do not.
# The state predicts the final payment status well across the entire range of credit ratings, and that is why it's a better predictor.
# Neither credit rating nor state are able to accurately predict default. That's why all the good regression algorithms I tried are always predicting "current"

xtabs(~ Payment.Status + Industry, data = cramerdfFinalPaymentStatus)
# Highly likely to pay in full (early!): "Accomodation and Food Services", "Information", "Professional, Scientific, and Technical Services", "Retail Trade"
# Likely to default (early!): none
xtabs(~ Payment.Status + Industry + Credit.rating, data = cramerdfFinalPaymentStatus)
# Same situation as State.

# Conclusion: State and Industry are better predictors of Payment Status than Credit Rating. Neither of them are good though.

# Now, see if there is a strong association between Credit.rating and Interest.rate
reg = lm(Interest.rate ~ Credit.rating, data = dfFinalPaymentStatus)
(R = sqrt(summary(reg)$r.squared)) # correlation between the observed durations, and the ones predicted (fitted) by our model
# Very closely related, indicating that Interest.rate was constructed from Credit.rating
# Conclude that the interest rate is really bad and could be improved upon.

# Do the same for Principal
reg = lm(Principal ~ Credit.rating, data = dfFinalPaymentStatus)
(R = sqrt(summary(reg)$r.squared)) # correlation between the observed durations, and the ones predicted (fitted) by our model
# The principal should not depend only on the credit rating because there are other factors (revenue, profit, etc.) involved. 
# Nonetheless, since the rest of the model (credit rating, interest rate) seems to be really bad, it's reasonable to think that the Principal is bad as well.





