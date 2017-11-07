require('lattice')
require('ggplot2')
require('methods')
require('caret') # confusionMatrix
require('ROCR') # ROC curve
require('e1071')   # SVM model
library('binr')
require('plyr')
require('rpart')
library('rpart.plot')
require('tree')
require('randomForest')


normalize <- function(x) {
    return ((x - min(x)) / (max(x) - min(x)))
  }


df = read.csv("titanic3.csv", na.strings=c("", "NA"))
# this data set has 1309 rows


# df$survived[which(df$survived==0)] <- "Died"
# df$survived[which(df$survived==1)] <- "Survived"
df$survived <- as.factor(df$survived)
# this data set is missing 3855 values
missing_attribute_count <- sum(is.na(df))

# Split into train and test set
smp_size <- floor(0.80 * nrow(df))
set.seed(1)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train_data <- df[train_ind, ]
test_data <- df[-train_ind, ]

# Fill in the missing age data with mean of male and female ages in train and test set seperately
train_data$age[which(is.na(train_data$age) & train_data$sex=="female")] <- mean(train_data$age[which(train_data$sex=="female")], na.rm = TRUE)
train_data$age[which(is.na(train_data$age) & train_data$sex=="male")] <- mean(train_data$age[which(train_data$sex=="male")], na.rm = TRUE)
test_data$age[which(is.na(test_data$age) & test_data$sex=="female")] <- mean(test_data$age[which(test_data$sex=="female")], na.rm = TRUE)
test_data$age[which(is.na(test_data$age) & test_data$sex=="male")] <- mean(test_data$age[which(test_data$sex=="male")], na.rm = TRUE)

# Fill in missing embarked data with S, which is the most frequent value
train_data$embarked[which(is.na(train_data$embarked))] <- "S"
test_data$embarked[which(is.na(test_data$embarked))] <- "S"

# Only keep pclass, age, sex, sibsp, parch, fare, embarked
train_data <- train_data[ -c(3, 8, 10, 12, 13, 14) ]
test_data <- test_data[ -c(3, 8,  10, 12, 13, 14) ]


train_data$age <- as.numeric(train_data$age)
test_data$age <- as.numeric(test_data$age)


# Bin Age and smooth by median
train_data["agebin"] <- NA
test_data["agebin"] <- NA
bins = c(0.092,8.15,16.1,24.1,32.1,40.1,48.1,56.1,64,72,80.1)
binss = c(0.092,4.16,8.15,12.1,16.1,20.1,24.1,28.1,32.1,36.1,40.1,44.1,48.1,52.1,56.1,60,64,68,72,76,80.1)

train_data$agebin <- .bincode(train_data$age,binss, TRUE,TRUE)
test_data$agebin <- .bincode(test_data$age,binss, TRUE,TRUE)


for(i in 1:20){
  train_data$age[which(train_data$agebin==i)]<- median(train_data$age[which(train_data$agebin==i)], na.rm = TRUE)
  test_data$age[which(test_data$agebin==i)]<- median(test_data$age[which(test_data$agebin==i)], na.rm = TRUE)

}
train_data$fare[which(is.na(train_data$fare))] <- median(train_data$fare, na.rm = TRUE)

#---------------------------------------------------------------------#
#----------------------------Decision Tree----------------------------#
#---------------------------------------------------------------------#
decision_tree <- rpart(survived ~.,data = train_data, cp=0.02, method="class")
rpart.plot(decision_tree)
varImp(decision_tree)

tree <- tree(survived ~.,train_data, wts=TRUE)
tree
plot(tree, type=c("uniform"))
text(tree,pretty = 1, label = "yval")


prunedCVTree <- cv.tree(tree, FUN=prune.misclass, K = 10)
plot(prunedCVTree)
bestsize <- prunedCVTree$size[which(prunedCVTree$dev==min(prunedCVTree$dev))]
bestsize[1]

prunedTree <- prune.misclass(tree, best=bestsize[1])
# summary(prunedTree)
prediction <- predict(prunedTree , test_data, type="class")
confMatrix <- confusionMatrix(prediction, test_data$survived, dnn=c("Prediction", "Reference"))

# Build ROC curve and AUC and find the pruned tree
pred <- prediction(as.numeric(prediction), as.numeric(test_data$survived))
roc <- performance(pred, "tpr", "fpr")

plot(roc, lwd=2, colorize=TRUE)
title(main="ROC Curve of Pruned Decision Tree")
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)
auc <- performance(pred, "auc")
auc = unlist(auc@y.values)
print(c(area_under_curve=auc))

#---------------------------------------------------------------------#
#----------------------------Random Forest----------------------------#
#---------------------------------------------------------------------#

forest <-randomForest(survived ~.,train_data, ntree=125 , importance=TRUE,
proximity=TRUE)

prediction <- predict(forest , test_data, type="class")

confMatrix <- confusionMatrix(prediction, test_data$survived, dnn=c("Prediction", "Reference"))
confMatrix

# Build ROC curve and AUC and find the pruned tree
pred <- prediction(as.numeric(prediction), as.numeric(test_data$survived))
roc <- performance(pred, "tpr", "fpr")

plot(roc, lwd=2, colorize=TRUE)
title(main="ROC Curve of Random Forest")
lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)
auc <- performance(pred, "auc")
auc = unlist(auc@y.values)
print(c(area_under_curve=auc))
importance(forest)
varImpPlot(forest)
