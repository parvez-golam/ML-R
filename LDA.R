#----------------------------------------------------------------------------- 
###                 LDA (Linear Discriminant Analysis)
#-----------------------------------------------------------------------------
# Iris data

library(MASS)

#-----------------------------------------------------------------------------
##-----------------------2-class classification problem
#-----------------------------------------------------------------------------

dat = iris
head(dat)

dat$Species = as.factor(ifelse(iris$Species=="virginica",1,0))
levels(dat$Species) = c("other","virginica")

par(mfcol=c(1,2))
hist(dat[which(dat$Species=="other"),1] , main="Histogram-other" )
hist(dat[which(dat$Species =="virginica"),1], main="Histogram-virginica")
par(mfcol=c(1,1))

set.seed(4061)

n = nrow(dat)
y = dat$Species
 
# shuffle dataset
dat = dat[sample(1:n),] 

# spliting train-test data
i.train = 1:100
dat.train = dat[i.train,]
dat.test = dat[-i.train,]
y.test = y[-i.train]

# check whether samples are representative of the overall dataset
table(dat$Species)
table(dat.train$Species)/100
table(dat.test$Species)/50

# Fit LDA
lda.o = lda(Species~., data=dat.train)
# Predict LDA
lda.p = predict(lda.o, newdata=dat.test)
# overall accuracy LDA
(tb = table(lda.p$class, y.test)) #dat.test$Species
acc.LDA = sum(diag(tb))/sum(tb)

# Fit QDA:
qda.o = qda(Species~., data=dat.train)
# Preict QDA
qda.p = predict(qda.o, newdata=dat.test)
# overall accuracy QDA
(tb = table(qda.p$class, y.test)) #dat.test$Species
acc.QDA = sum(diag(tb))/sum(tb)


#----------------------------------------------------------------
# perform K-fold CV:

set.seed(4061)

n = nrow(dat)
y = dat$Species

K = 10
acc.LDA.cv =  numeric(K)
acc.QDA.cv =  numeric(K)

folds = cut(1:n, K, labels=FALSE)
for(k in 1:K){
	#WITHout kth dataset
	i.train = which(folds!=k)
	dat.train = dat[i.train, ]
	dat.test = dat[-i.train, ]
	y.test = y[-i.train]

	# Fit LDA
	lda.o = lda(Species~., data=dat.train)
	# Predict LDA
	lda.p = predict(lda.o, newdata=dat.test)
	# accuracy LDA
	tb = table(lda.p$class, y.test)    #dat.test$Species
	acc.LDA.cv[k] = sum(diag(tb))/sum(tb)

	# Fit QDA:
	qda.o = qda(Species~., data=dat.train)
	# Preict QDA
	qda.p = predict(qda.o, newdata=dat.test)
	# accuracy QDA
	tb = table(qda.p$class, y.test)    #dat.test$Species
	acc.QDA.cv[k] = sum(diag(tb))/sum(tb)
}	

acc.LDA.cv
# overall accurecy LDA
mean(acc.LDA.cv)

acc.QDA.cv
# overall accurecy QDA
mean(acc.LDA.cv)



#--------------------------------------------------------------------------
## Bootstrap --- OOB evaluation

set.seed(4061)
n = nrow(dat)
y = dat$Species

B = 100
acc.LDA.oob = numeric(B)
acc.QDA.oob = numeric(B)

for(b in 1:B){
	ib = sample(1:n, n, replace=TRUE)
	dat.train = dat[ib,]
	dat.test = dat[-ib,]
	y.test = y[-ib]

	# Fit LDA
	lda.o = lda(Species~., data=dat.train)
	# Predict LDA
	lda.p = predict(lda.o, newdata=dat.test)
	# accuracy LDA
	tb = table(lda.p$class, y.test)    #dat.test$Species
	acc.LDA.oob[b] = sum(diag(tb))/sum(tb)

	# Fit QDA:
	qda.o = qda(Species~., data=dat.train)
	# Preict QDA
	qda.p = predict(qda.o, newdata=dat.test)
	# accuracy QDA
	tb = table(qda.p$class, y.test)    #dat.test$Species
	acc.QDA.oob[b] = sum(diag(tb))/sum(tb)
}

acc.LDA.oob
# overall classification accuracy rate LDA
mean(acc.LDA.oob)

acc.QDA.oob
# overall classification accuracy rate QDA
mean(acc.QDA.oob)

boxplot( acc.LDA.oob, acc.LDA.cv, acc.QDA.oob, acc.QDA.cv,
	ylim=c(0.4,1.3), 
	main="Different classification accuracy estimates",
	names=c("LDA-OOB", "LDA-CV-10", "QDA-OOB", "QDA-CV-10"))
abline(h= acc.LDA, col="blue")
abline(h= acc.QDA, col="red")



#-----------------------------------------------------------------------------
## 3-class classification problem
#-----------------------------------------------------------------------------

dat = iris
n = nrow(dat)

set.seed(4061)
# siffle data
dat = dat[sample(1:n),]

i.train = 1:100
dat.train = dat[i.train,]
dat.test = dat[-i.train,]

# LDA:
lda.o = lda(Species~., data=dat.train)
lda.p = predict(lda.o, newdata=dat.test)
names(lda.p)
(tb = table(lda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)

# QDA:
qda.o = qda(Species~., data=dat.train)
qda.p = predict(qda.o, newdata=dat.test)
(tb = table(qda.p$class, dat.test$Species))
sum(diag(tb))/sum(tb)


#----------------------------------------------------------------
# perform K-fold CV:

set.seed(4061)

dat = iris
n = nrow(dat)
y = dat$Species

K = 10
acc.LDA.cv =  numeric(K)
acc.QDA.cv =  numeric(K)

folds = cut(1:n, K, labels=FALSE)
for(k in 1:K){
	#WITHout kth dataset
	i.train = which(folds!=k)
	dat.train = dat[i.train, ]
	dat.test = dat[-i.train, ]
	y.test = y[-i.train]

	# Fit LDA
	lda.o = lda(Species~., data=dat.train)
	# Predict LDA
	lda.p = predict(lda.o, newdata=dat.test)
	# accuracy LDA
	tb = table(lda.p$class, y.test)    #dat.test$Species
	acc.LDA.cv[k] = sum(diag(tb))/sum(tb)

	# Fit QDA:
	qda.o = qda(Species~., data=dat.train)
	# Preict QDA
	qda.p = predict(qda.o, newdata=dat.test)
	# accuracy QDA
	tb = table(qda.p$class, y.test)    #dat.test$Species
	acc.QDA.cv[k] = sum(diag(tb))/sum(tb)
}	

acc.LDA.cv
# overall accurecy LDA
mean(acc.LDA.cv)

acc.QDA.cv
# overall accurecy QDA
mean(acc.LDA.cv)



#--------------------------------------------------------------------------
## Bootstrap --- OOB evaluation

set.seed(4061)

dat = iris
n = nrow(dat)
y = dat$Species

B = 100
acc.LDA.oob = numeric(B)
acc.QDA.oob = numeric(B)

for(b in 1:B){
	ib = sample(1:n, n, replace=TRUE)
	dat.train = dat[ib,]
	dat.test = dat[-ib,]
	y.test = y[-ib]

	# Fit LDA
	lda.o = lda(Species~., data=dat.train)
	# Predict LDA
	lda.p = predict(lda.o, newdata=dat.test)
	# accuracy LDA
	tb = table(lda.p$class, y.test)    #dat.test$Species
	acc.LDA.oob[b] = sum(diag(tb))/sum(tb)

	# Fit QDA:
	qda.o = qda(Species~., data=dat.train)
	# Preict QDA
	qda.p = predict(qda.o, newdata=dat.test)
	# accuracy QDA
	tb = table(qda.p$class, y.test)    #dat.test$Species
	acc.QDA.oob[b] = sum(diag(tb))/sum(tb)
}

acc.LDA.oob
# overall classification accuracy rate LDA
mean(acc.LDA.oob)

acc.QDA.oob
# overall classification accuracy rate QDA
mean(acc.QDA.oob)

boxplot( acc.LDA.oob, acc.LDA.cv, acc.QDA.oob, acc.QDA.cv,
	ylim=c(0.4,1.3), 
	main="Different classification accuracy estimates",
	names=c("LDA-OOB", "LDA-CV-10", "QDA-OOB", "QDA-CV-10"))
abline(h= acc.LDA, col="blue")
abline(h= acc.QDA, col="red")






