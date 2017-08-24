## SYS 6018
## Data Mining Homework 1
## Yi Hao, yh8a

library(readr)  
library(dplyr)  
library(ggplot2)

train <- read_csv("train.csv")
summary(train) # in train dataset, variable "Age" has 177 NA's

test <- read_csv("test.csv")
summary(test) #in test dataset, variable "Age" has 86 NA's and variable "Fare" has 1 NA

#mean imputation
train1 <- train
train1$Age[is.na(train1$Age)] <- mean(train1$Age[!is.na(train1$Age)])
summary(train1)

test1 <- test
test1$Age[is.na(test1$Age)] <- mean(train1$Age[!is.na(train1$Age)]) # use mean value of the train set to impute the NA in the test set
test1$Fare[is.na(test1$Fare)] <- mean(train1$Fare[!is.na(train1$Fare)])
summary(test1)

table(train1$Parch)
table(test1$Parch)
# Parch >= 3 are very rare cases, so they can be combined into one level
transform_parch <- function(x) {
  if (x==0) {
    return("0")
  }
  else if (x==1) {
    return("1")
  }
  else if (x==2) {
    return ("2")
  }
  else if (x>=3) {
    return ("3+")
  }
}

train1$Parch <- sapply(train1$Parch, transform_parch)
test1$Parch <- sapply(test1$Parch, transform_parch)

# Lots of categorical variables
train1$Pclass <- factor(train1$Pclass)
train1$Sex <- factor(train1$Sex)
train1$SibSp <- factor(train1$SibSp)
train1$Parch <- factor(train1$Parch)
train1$Embarked <- factor(train1$Embarked)

test1$Pclass <- factor(test1$Pclass)
test1$Sex <- factor(test1$Sex)
test1$SibSp <- factor(test1$SibSp)
test1$Parch <- factor(test1$Parch)
test1$Embarked <- factor(test1$Embarked)

# Now we select a random sample for training and cross-validation
dim(train1)
s2 <- sample(1:891, size=445) 
tr1.train <- train1[s2,]
tr1.valid <- train1[-s2,]

# Let's start with the full model including all variables minus Name, Cabin and Ticket
survival.lg1 <- glm(Survived~Pclass+Sex+SibSp+Parch+Embarked+Age+Fare, data=tr1.train, family = "binomial")
summary(survival.lg1)

# We'll just use p > 0.5 for a threshold
preds <- rep(0,445)  # Initialize prediction vector
preds[survival.lg1$fitted.values>0.5] <- 1 # p>0.5 -> 1
table(preds,tr1.train$Survived)
# 65.8% correct

#Let's look at the ANOVA output, to try to identify a subset of useful variables.
anova(survival.lg1,test="Chisq")

# Only Pclass, Sex, SibSp, Age and Embarked seem to add something to the model.
# Let's try just those variables.
survival.lg2 <- glm(Survived~Pclass+Sex+SibSp+Age+Embarked, data=tr1.train, family = "binomial")
summary(survival.lg2)

#AIC went down
#Let's look at ANOVA for this subset 
anova(survival.lg2,test="Chisq")

# Mostly significant, let's see how this predicts.
preds <- rep(0,445)  # Initialize prediction vector
preds[survival.lg2$fitted.values>0.5] <- 1 # p>0.5 -> 1
table(preds,tr1.train$Survived)
# This time we got 65.2% correct.

# One more model, with Embarked removed:
survival.lg3 <- glm(Survived~Pclass+Sex+SibSp+Age, data=tr1.train, family = "binomial")
summary(survival.lg3)
anova(survival.lg3,test="Chisq")
# AIC about the same as 2nd model, ANOVA says all variables significant.
# Predictions:
preds <- rep(0,445)  # Initialize prediction vector
preds[survival.lg2$fitted.values>0.5] <- 1 # p>0.5 -> 1
table(preds,tr1.train$Survived)
# This time we got 65.2%, same as second model.

# Now let's cross-validate with both models.

# Model 1: All variables except Name, Cabin and Ticket
probs<-as.vector(predict(survival.lg1,newdata=tr1.valid, type="response"))
preds <- rep(0,446)  # Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
table(preds,tr1.valid$Survived)
# Correct 76.7

# Model 2: Include Pclass, Sex, SibSp, Age and Embarked.
probs<-as.vector(predict(survival.lg2,newdata=tr1.valid, type="response"))
preds <- rep(0,446)  # Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
table(preds,tr1.valid$Survived)
# Correct 76.7%  

# Model 3: Include Pclass, Sex, SibSp, Age.
probs<-as.vector(predict(survival.lg3,newdata=tr1.valid, type="response"))
preds <- rep(0,446)  # Initialize prediction vector
preds[probs>0.5] <- 1 # p>0.5 -> 1
table(preds,tr1.valid$Survived)
# Correct 77.4%  

# By comparason, preferred -- model 3

# Now for the predictions:
survival.lg <- glm(Survived~Pclass+Sex+SibSp+Age, data=train1, family = "binomial")

probs<-as.vector(predict(survival.lg,newdata=test1, type="response"))
pred2 <- rep(0,418)  # Initialize prediction vector
pred2[probs>0.5] <- 1 # p>0.5 -> 1
pred2 <- data.frame(pred2)
pred2$PassengerId <- test1$PassengerId
names(pred2) <- c("Survived","PassengerId")
pred2<-pred2[,c(2,1)]
write_csv(pred2,"yh8a_titanic_predictions.csv")



