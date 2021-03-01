
#--------------------------------------------------------------------------
#                        Logistic Regression
#--------------------------------------------------------------------------
# iris data

set.seed(4061)

n = nrow(iris)
is = sample(1:n, size=n, replace=FALSE)

# suffle data
# only Sepal information 
dat = iris[is,-c(3,4)]
head(dat)

# recode into 2-class problem:
dat$is.virginica = as.numeric(dat$Species=="virginica") 
dat$Species = NULL # "remove" this component
names(dat)

x = dat[,1:2]
y = dat$is.virginica

# split train-test data
i.train = is[1:100] 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]

# logistic regression
#fit model
fit = glm(is.virginica~., data=dat, subset=i.train, family=binomial(logit))

# predict from model
pred = predict(fit, newdata=dat[-i.train,], type="response")

boxplot(pred~y.test, names=c("other","virginica"))
abline(h=0.5, col=3)
abline(h=0.1, col=4)

# over all error rate
pred.y = as.numeric(pred>0.5)
tb = table(pred.y, y.test)
err.rate = 1-sum(diag(tb))/sum(tb)



# for varying cut-off (ie threshold) values, compute corresponding 
# predicted labels, and corresponding confusion matrix:
for(cut.off in seq(.1, .9, by=.1)){
	pred.y = as.numeric(pred>cut.off)
	tb = table(pred.y, y.test)
	print(1-sum(diag(tb))/sum(tb))
}


#----------------------------------------------------------------
# perform K-fold CV:

set.seed(4061)
K = 10
n = nrow(dat)

x = dat[,1:2]
y = dat$is.virginica

acc.glm =  numeric(K)

folds = cut(1:n, K, labels=FALSE)
for(k in 1:K){
	#WITHout kth dataset
	i.train = which(folds!=k)
	dat.train = dat[i.train, ]
	dat.test = dat[-i.train, ]
	y.test = y[-i.train]

	# logestic regression
	glm.o = glm(is.virginica~., data=dat.train, family=binomial(logit))

	#Predict/Test model with kth fold dataset
	glm.p = ( predict(glm.o, newdata=dat.test, type="response") > 0.5 )
	tb.glm = table(glm.p, y.test)

	#prediction accuracies
	acc.glm[k] = sum(diag(tb.glm)) / sum(tb.glm)
}	

acc.glm
# overall accurecy
mean(acc.glm)


#--------------------------------------------------------------------------
## Bootstrap --- OOB evaluation

set.seed(4061)
n = nrow(dat)

B = 100
acc.glm.oob = numeric(B)

for(b in 1:B){
	ib = sample(1:n, n, replace=TRUE)
	dat.train = dat[ib,]
	dat.test = dat[-ib,]
	y.test = y[-ib]

	#logestic regression
	glm.o = glm(is.virginica~., data=dat.train, family=binomial(logit))
	
	#Predict/Test model with kth fold dataset
	glm.p = ( predict(glm.o, newdata=dat.test, type="response") > 0.5 )
	tb.glm = table(glm.p, y.test)

	#prediction accuracies
	acc.glm.oob[b] = sum(diag(tb.glm)) / sum(tb.glm)
}

acc.glm.oob
# overall classification accuracy rate
mean(acc.glm.oob)

boxplot( acc.glm.oob, acc.glm,
	ylim=c(0,1), 
	main="Different classification accuracy estimates",
	names=c("OOB", "CV-10"))
abline(h=(1-err.rate), col="blue")



