#--------------------------------------------------------------------------
#          Comparing regressions ( LInear reg(OLS), Ridge, LASSO )
#---------------------------------------------------------------------------
# "Prestige" dataset

library(glmnet)
library(car)

head(Prestige)

set.seed(4061)

y = Prestige$income
x = Prestige[,c("education","prestige","women")]

#Train test split
set.seed(1)
itrain = sample(c(1:nrow(x)), round(.7*nrow(x))) # 70%-30% split
x.train = x[itrain,]
y.train = y[itrain]
x.test = x[-itrain,]
y.test = y[-itrain]


#---------------------LInear regression
#Fit model on Prestige data
lmo = lm(y.train~., data=x.train) 
summary(lmo)

# predict from training data
lmo.train.pred = predict(lmo, newx=x.train)
lmo.train.sqr.err = (lmo.train.pred - y.train)^2

# predict from test data
lmo.pred = predict(lmo, newdata=x.test)
lmo.test.sqr.err = (lmo.pred- y.test)^2


#----------------------LASSO
xm.train = model.matrix(y.train~., data=x.train)[,-1]
xm.test = model.matrix(y.test~., data=x.test)[,-1]

#Fit model on Prestige data
lasso.cv = cv.glmnet(xm.train, y.train, alpha=1)
lasso = glmnet(xm.train, y.train, alpha=1, lambda=lasso.cv$lambda.min)

# predict from training data
lasso.train.pred = predict(lasso, newx=xm.train)
lasso.train.sqr.err = (lasso.train.pred - y.train)^2

# predict from test data
lasso.pred = predict(lasso, newx=as.matrix(x.test))
lasso.test.sqr.err = (lasso.pred - y.test)^2


#-----------------------Ridge
xm.train = model.matrix(y.train~., data=x.train)[,-1]
xm.test = model.matrix(y.test~., data=x.test)[,-1]

#Fit model on Prestige data
ridge.cv = cv.glmnet(xm.train, y.train, alpha=0)
ridge = glmnet(xm.train, y.train, alpha=0, lambda=ridge.cv$lambda.min)

# predict from training data
ridge.train.pred = predict(ridge, newx=xm.train)
ridge.train.sqr.err = (ridge.train.pred - y.train)^2

# predict from test data
ridge.pred = predict(ridge, newx=as.matrix(x.test))
ridge.test.sqr.err = (ridge.pred - y.test)^2



cbind(coef(lmo), coef(lasso), coef(ridge))

# prediction comparision
cbind(lmo.pred, lasso.pred, ridge.pred)

boxplot( lmo.train.sqr.err, lmo.test.sqr.err,
	 ridge.train.sqr.err, ridge.test.sqr.err,
	lasso.train.sqr.err, lasso.test.sqr.err,
	ylim = c(0,10^7),
	main = "Different Squraed Error comparision",
	names = c("Lmo-train", "Lmo-test",
		    "Ridge-train", "Ridge-test",
		    "LASSO-train", "LASSO-test"),
	col = c('pink', 'pink', 'cyan', 'cyan', 'blue', 'blue'))










