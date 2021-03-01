#----------------------------------------------------------------------------
##                           Tree-based methods
#----------------------------------------------------------------------------
# Carseats dataset

library(ISLR) # contains the dataset
library(tree) # contains... tree-building methods


y = ifelse(Carseats$Sales<=8, "No", "Yes")
dat = data.frame(Carseats, y)
dat$Sales <- NULL
dat$y = as.factor(dat$y)


# Fit tree to response variable y
tree.out = tree(y~., dat)
summary(tree.out)

# plot the tree
plot(tree.out)
text(tree.out, pretty=0)


#----------------------------------------------
#------------------------ pruning 

set.seed(3)
cv.dat = cv.tree(tree.out, FUN=prune.misclass)

names(cv.CS)
# - size:
# number of terminal nodes in each tree in the cost-complexity 
# pruning sequence.
# - deviance:	
# total deviance of each tree in the cost-complexity pruning sequence.
# - k:
# the value of the cost-complexity pruning parameter of each tree 
# in the sequence.

cv.dat

par(mfrow=c(1,2))
plot(cv.dat$size,cv.dat$dev,t='b')
plot(cv.dat$k,cv.dat$dev,t='b')



# use pruning: 
# - use which.min(cv.dat$dev) to get the location of the optimum
# - retrieve the corresponding tree size
# - pass this information on to pruning function

opt.size = cv.dat$size[which.min(cv.dat$dev)]
ptree = prune.misclass(tree.out, best=opt.size)
ptree 
summary(ptree)


par(mfrow=c(1,2))
plot(tree.out)
text(tree.out, pretty=0)
plot(ptree)
text(ptree, pretty=0)

par(mfrow=c(1,1))


#-----------------------------------------------------------------
#-----------------apply CV and ROC analysis

# train/test data split:
set.seed(4061)
n = nrow(dat)
itrain = sample(1:n, round(n*0.5))
dat.test = dat[-itrain,]
y.test = y[-itrain]

# Fitting a tree with training set
tree.out = tree(y~., dat, subset=itrain)
summary(tree.out)

plot(tree.out)
text(tree.out, pretty=0)


# prediction from full tree with test dataset:
tree.pred = predict(tree.out, dat.test, type="class")
# confusion matrix
(tb1 = table(tree.pred, y.test))  

# Accuracy
sum(diag(tb1))/sum(tb1)

# prune tree and get corresponding predictions:
cv.dat = cv.tree(tree.out, FUN=prune.misclass)
opt.size = cv.dat$size[which.min(cv.dat$dev)]

ptree = prune.misclass(tree.out, best=opt.size)
ptree.pred = predict(ptree, dat.test, type="class")
# confusion matrix
(tb2 = table(ptree.pred, y.test)) 

# Accuracy
sum(diag(tb2))/sum(tb2)


##------------------------------------------------------------
# perform ROC analysis

library(pROC)

# here we specify 'type="vector"' to retrieve continuous scores
# as opposed to predicted labels, so that we can apply varying
# threshold values to these scores to build the ROC curve:

# ROC for fully grown tree
tree.out.probs = predict(tree.out, dat.test, type="vector")
roc.tree = roc(response=(y.test), predictor=tree.out.probs[,1])
roc.tree$auc


# ROC for Purned tree
ptree.probs = predict(ptree, dat.test, type="vector")
roc.p = roc(response=(y.test), predictor=ptree.probs[,1])
roc.p$auc


plot(roc.p)
plot(roc.tree, col=4, add=TRUE)
legend("bottomright", bty='n', col=c(1,4), lty=1, lwd=3, 
	legend=c("Pruned","Fully grown"))

