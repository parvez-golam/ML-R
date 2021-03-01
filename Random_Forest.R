#---------------------------------------------------------------------------
##                         Random Forests
#---------------------------------------------------------------------------
# Carseats dataset

library(tree)
library(ISLR)
library(randomForest)


y = ifelse(Carseats$Sales <= 8, 'No', 'Yes')
CS = data.frame(Carseats, y)
CS$Sales = NULL
CS$y = as.factor(CS$y)


# train-test split
set.seed(6041)
N = nrow(CS)
itrain = sample(1:N, round(N*0.5))
CS.train = CS[itrain,] 
CS.test = CS[-itrain,] 


#----------------grow a single (unpruned) tree

tree.out = tree(y~., CS.train)
# fitted values for "train set"
tree.yhat = predict(tree.out, CS.train, type="class")
# fitted values for "test set"
tree.pred = predict(tree.out, CS.test, type="class")

# confusion matrix for tree (test data):
(tb.tree = table(tree.pred, CS.test$y))

# Accuracy
sum(diag(tb.tree))/sum(tb.tree)



#-----------------grow a forest:

rf.out = randomForest(y~., CS.train)

# fitted values for "training set"
rf.yhat = predict(rf.out, CS.train, type="class")
# fitted values for "test set"
rf.pred = predict(rf.out, CS.test, type="class")

# confusion matrix for RF (test data):
(tb.rf = table(rf.pred, CS.test$y))

# Accuracy
sum(diag(tb.rf))/sum(tb.rf)



#------------------compare to bagging:

bag.out = randomForest(y~., CS.train, mtry=(ncol(CS)-1))

# fitted values for "training set"
bag.yhat = predict(bag.out, CS.train, type="class")

# fitted values for "test set"
bag.pred = predict(bag.out, CS.test, type="class")

# confusion matrix for RF (test data):
(tb.bag = table(bag.pred, CS.test$y))

# Accuracy
sum(diag(tb.bag))/sum(tb.bag)







