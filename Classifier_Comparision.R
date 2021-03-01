#-----------------------------------------------------------------------------
## -      comparing classifiers(kNN, logistic regression, LDA and QDA)
#------------------------------------------------------------------------------

library(class) # contains knn()
library(MASS)  # contains LDA
library(ISLR) # contains the datasets
library(pROC) 

#---------------------------------------------------------------------
# Default dataset
#--------------------------------------------------------------------

set.seed(4061)
n = nrow(Default)
dat = Default[sample(1:n, n, replace=FALSE), ]

# Scaleing the dataset
my.scale <- function(x, ...){
	for(i in 1:ncol(x)){
		if(class(x[,i])!="factor"){
			x[,i] <- scale(x[,i], ...)
		}
	}
	return(x)
}
dat = my.scale(dat)

# get a random training sample containing 70% of original sample:
i.cv = sample(1:n, round(.7*n), replace=FALSE)
dat.cv = dat[i.cv,] # use this for CV (train+test)
dat.valid = dat[-i.cv,] # save this for later (after CV)

# tuning of the classifiers:
K.knn = 3 

# perform K-fold CV:
K = 10 
N = length(i.cv)
folds = cut(1:N, K, labels=FALSE)
acc.knn = acc.glm = acc.lda = acc.qda = numeric(K)

# for ROC analysis
auc.knn = auc.glm = auc.lda = auc.qda = numeric(K)

for(k in 1:K){ # 10-fold CV loop
	# split into train and test samples:
	i.train	= which(folds!=k)
	dat.train = dat.cv[i.train, ]
	dat.test = dat.cv[-i.train, ]

	# adapt these sets for kNN:
	x.train = dat.train[,-1]
	y.train = dat.train[,1]
	x.test = dat.test[,-1]
	y.test = dat.test[,1]
	x.train[,1] = as.numeric(x.train[,1])
	x.test[,1] = as.numeric(x.test[,1])
	
	# train and test knn:
	knn.o = knn(x.train, x.test, y.train, K.knn)
	knn.p = knn.o
	tb.knn = table(knn.p, y.test)
	
	# train and test logistic regression:
	glm.o = glm(default~., data=dat.train, family=binomial(logit))
	glm.p = ( predict(glm.o, newdata=dat.test, type="response") > 0.5 )
	tb.glm = table(glm.p, y.test)
	
	# train and test LDA
	lda.o = lda(default~., data=dat.train)
	lda.p = predict(lda.o, newdata=dat.test)$class
	tb.lda = table(lda.p, y.test)
	
	# train and test QDA
	qda.o = qda(default~., data=dat.train)
	qda.p = predict(qda.o, newdata=dat.test)$class	
	tb.qda = table(qda.p, y.test)
	
	# store prediction accuracies:
	acc.knn[k] = sum(diag(tb.knn)) / sum(tb.knn)
	acc.glm[k] = sum(diag(tb.glm)) / sum(tb.glm)
	acc.lda[k] = sum(diag(tb.lda)) / sum(tb.lda)
	acc.qda[k] = sum(diag(tb.qda)) / sum(tb.qda)
	
	# ROC/AUC analysis:
	# WARNING: THIS IS NOT PR(Y=1 | X), BUT Pr(Y = Y_hat | X):
	# knn.p = attributes(knn(x.train, x.test, y.train, K.knn, prob=TRUE))$prob
	glm.p = predict(glm.o, newdata=dat.test, type="response")
	lda.p = predict(lda.o, newdata=dat.test)$posterior[,2]
	qda.p = predict(qda.o, newdata=dat.test)$posterior[,2]
	# auc.knn[k] = roc(y.test, knn.p)$auc
	auc.glm[k] = roc(y.test, glm.p)$auc
	auc.lda[k] = roc(y.test, lda.p)$auc
	auc.qda[k] = roc(y.test, qda.p)$auc

}

#dev.new()
boxplot(acc.knn, acc.glm, acc.lda, acc.qda,
	main="Overall CV prediction accuracy",
	names=c("kNN","GLM","LDA","QDA"))
c( mean(acc.knn), mean(acc.glm), mean(acc.lda), mean(acc.qda) )


boxplot(auc.glm, auc.lda, auc.qda,
	main="Overall CV AUC",
	names=c("GLM","LDA","QDA"))
# boxplot(auc.knn, auc.glm, auc.lda, auc.qda,
	# main="Overall CV AUC",
	# names=c("kNN","GLM","LDA","QDA"))

