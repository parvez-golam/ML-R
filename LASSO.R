#--------------------------------------------------------------------------
#-------------------------LASSO regression
#--------------------------------------------------------------------------
# Hitters data set

library(ISLR)
library(glmnet)

set.seed(4061)

#Remove 'NA' 
dat = na.omit(Hitters)
# suffle data
dat = dat[sample(1:n, n, replace=FALSE),]

n = nrow(dat)
x = model.matrix(Salary~., data=dat)[,-1]
y = dat$Salary

# Train-test split
i.train = sample(c(1:n), round(.7*n)) #70%-30% split
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]


# lambda with cross validation
#----------------------------------------------------------------

# set alpha=1 for LASSO
lamda.cv = cv.glmnet(x.train, y.train, alpha=1) 
lamda.cv$lambda.min

# model fit on training data with
lasso.mod.cv = glmnet(x.train, y.train, alpha=1, lambda=lamda.cv$lambda.min)

# Model coefficients
round(coef(lasso.mod.cv),3)

# predictions on test data
lasso.pred.cv = predict(lasso.mod.cv, newx=x.test)

# compute RMSE for the test data:
rmse1 = sqrt(mean((lasso.pred.cv-y.test)^2))




# lambda value 10
#----------------------------------------------------------------

# model fit on training data
lasso.mod = glmnet(x.train, y.train, alpha=1, lambda=10)

# coefficients
round(coef(lasso.mod),3)

# predictions on test data
lasso.pred = predict(lasso.mod, newx=x.test)


# compute RMSE for the test data:
rmse2 = sqrt(mean((lasso.pred-y.test)^2))




# Cross validation for Train and test
#---------------------------------------------------------------------
# perform K-fold CV:

set.seed(4061)

n = nrow(dat)
x = model.matrix(Salary~., data=dat)[,-1]
y = dat$Salary

K = 10

rmse.cv = numeric(K)

folds = cut(1:n, K, labels=FALSE)
for(k in 1:K){
	#WITHout kth dataset
	i.train = which(folds!=k)
	x.train = x[i.train,]
	x.test = x[-i.train,]
	y.train = y[i.train]
	y.test = y[-i.train]

	# set alpha=1 for LASSO
	lamda.cv = cv.glmnet(x.train, y.train, alpha=1) 

	# model fit on training data with
	lasso.mod1 = glmnet(x.train, y.train, alpha=1, lambda=lamda.cv$lambda.min)

	# predictions on test data
	lasso.pred1 = predict(lasso.mod1, newx=x.test)

	# compute RMSE for the test data:
	rmse.cv[k] = sqrt(mean((lasso.pred1-y.test)^2))
}	

rmse.cv
# RMSE
mean(rmse.cv)


#--------------------------------------------------------------------------
## Bootstrap --- OOB evaluation

set.seed(4061)
n = nrow(dat)
x = model.matrix(Salary~., data=dat)[,-1]
y = dat$Salary

B = 100
rmse.oob = numeric(B)

for(b in 1:B){
	ib = sample(1:n, n, replace=TRUE)
	#uib = unique(ib)
	x.train = x[ib,]
	x.test = x[-ib,]
	y.train = y[ib]
	y.test = y[-ib]

	# set alpha=1 for ridge regression
	lamda.cv = cv.glmnet(x.train, y.train, alpha=1) 

	# model fit on training data with
	lasso.mod2 = glmnet(x.train, y.train, alpha=1, lambda=lamda.cv$lambda.min)

	# predictions on test data
	lasso.pred2 = predict(lasso.mod2, newx=x.test)

	# compute RMSE for the test data:
	rmse.oob[b] = sqrt(mean((lasso.pred2-y.test)^2))	
}

rmse.oob
# RMSE
mean(rmse.oob)


boxplot(rmse.oob, rmse.cv, 
	main="Different RMSE estimates for LASSO",
	names=c("OOB", "CV-10"))
abline(h=rmse1, col="blue")
abline(h=rmse2, col="red")

