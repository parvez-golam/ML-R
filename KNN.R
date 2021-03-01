
#--------------------------------------------------------------------------
#                        KNN
#--------------------------------------------------------------------------
# iris data

library(class)

set.seed(4061)

# shuffle data
z = iris[sample(1:nrow(iris)),]
head(z)

#only Sepal information
plot(z[,1:2], col=c(1,2,4)[z[,5]], pch =20, cex= 2)

x = z[,c(1,2)]
y = z$Species
head(x)



k = 5
n = nrow(x)

# split train-test data
i.train = sample(1:n, 100)
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]

# KNN classifier
ko = knn(x.train, x.test, y.train, k)
tb = table(ko, y.test)

# overall classification error rate
err.rate = 1 - sum(diag(tb)) / sum(tb) 


#-Confusion matrix
library(caret)
confusionMatrix(data=ko, reference=y.test)


# Build a loop around that to find best k:
# (NB: assess effect of various k-values 
# on the same data-split)
Kmax = 30
acc = numeric(Kmax)
for(k in 1:Kmax){
	ko = knn(x.train, x.test, y.train, k)
	tb = table(ko, y.test)
	acc[k] = sum(diag(tb)) / sum(tb)	
}
acc
plot(1-acc, pch=20, t='b', xlab='k')


#LOO-CV---------------------------------------------------------------
set.seed(4061)
k = 5
n = nrow(x)

err.LOO = numeric(n)
for(i in 1:n){
	# WITHout ith dataset
	i.train = sample(1:n)[-i]
	x.train = x[i.train,]
	x.test = x[-i.train,]
	y.train = y[i.train]
	y.test = y[-i.train]

	ko = knn(x.train, x.test, y.train, k)
	
	#Test/predict with ith dataset
	tb = table(ko, y.test)
	
	#classification error rate for ith case
	err.LOO[i] = 1 - sum(diag(tb)) / sum(tb)
}

err.LOO
# overall classification error rate
mean(err.LOO)

#-------------------------------------------------------------------- 
#5-fold cross-validation

set.seed(4061)
knn = 5
n = nrow(x)

K = 5
err.K5 = numeric(K)
folds = cut(1:n, K, labels=FALSE)

for(k in 1:K){
  	#WITHout kth dataset
	i.train = which(folds!=k)
	x.train = x[i.train,]
	x.test = x[-i.train,]
	y.train = y[i.train]
	y.test = y[-i.train]

	ko = knn(x.train, x.test, y.train, knn)
	
	#Predict/Test with kth fold dataset
	tb = table(ko, y.test)
	
	#classification error rate for kth fold
	err.K5[k] = 1 - sum(diag(tb)) / sum(tb)
}

err.K5
# overall classification error rate
mean(err.K5)


#-------------------------------------------------------------------- 
#10-fold cross-validation

set.seed(4061)
knn = 5
n = nrow(x)

K = 10
err.K10 = numeric(K)
folds = cut(1:n, K, labels=FALSE)

for(k in 1:K){
  	#WITHout kth dataset
	i.train = which(folds!=k)
	x.train = x[i.train,]
	x.test = x[-i.train,]
	y.train = y[i.train]
	y.test = y[-i.train]

	ko = knn(x.train, x.test, y.train, knn)
	
	#Predict/Test model with kth fold dataset
	tb = table(ko, y.test)
	
	#classification error rate for kth fold
	err.K10[k] = 1 - sum(diag(tb)) / sum(tb)
}

err.K10
# overall classification error rate
mean(err.K10)


# error plots
boxplot(err.LOO, err.K5, err.K10, 
	main="Different CV error estimates",
	names=c("LOO-CV","K=5","K=10"))
abline(h=err.rate)


#--------------------------------------------------------------------------
## Bootstrap --- OOB evaluation

set.seed(4061)
knn = 5
n = nrow(x)

B = 100
err.OOB = numeric(B)

for(b in 1:B){
	ib = sample(1:n, n, replace=TRUE)
	#uib = unique(ib)
	x.train = x[ib,]
	x.test = x[-ib,]
	y.train = y[ib]
	y.test = y[-ib]

	ko = knn(x.train, x.test, y.train, knn)
	
	tb = table(ko, y.test)
	err.OOB[b] = 1 - sum(diag(tb)) / sum(tb)	
}

err.OOB
# overall classification error rate
mean(err.OOB)

boxplot(err.LOO, err.K5, err.OOB,err.K10,
	ylim=c(0,1), 
	main="Different classification error estimates",
	names=c("CV-LOO","CV-5","OOB", "CV-10"))
abline(h=err.rate, col="blue")



