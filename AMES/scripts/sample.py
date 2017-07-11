import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.model_selection import cross_val_score
import matplotlib
import matplotlib.pyplot as plt
pd.options.display.max_columns = 300

train = pd.read_csv("../data/train.csv")
target = train["SalePrice"]
train = train.drop("SalePrice",1) # take out the target variable
test = pd.read_csv("../data/test.csv")
combi = pd.concat((train,test)) # this is the combined data frame without the target variable

print(np.shape(train))
print(np.shape(test))
print(np.shape(combi))

# figure(figsize(8,4))
# subplot(1,2,1)
# hist(target*1e-6,20);
# xlabel("Sale Price in Mio Dollar")
# subplot(1,2,2)
# hist(log10(target),20);
# xlabel("log10(Sale Price in Dollar)")

target = np.log10(target)
# find missing features
total = combi.isnull().sum().sort_values(ascending=False)
percent = (combi.isnull().sum()/combi.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(20))
# create new features from categorical data:
combi = combi.drop((missing_data[missing_data['Total'] > 1]).index,1)
combi = combi.drop(combi.loc[combi['Electrical'].isnull()].index)
print(combi.isnull().sum().max()) #just checking that there's no missing data missing...combi = pd.get_dummies(combi)
# and fill missing entries with the column mean:
combi = pd.get_dummies(combi)
combi = combi.fillna(combi.mean())

# create the new train and test arrays:
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]

combi.head(10)

model = LinearRegression()
score = np.mean(np.sqrt(-cross_val_score(model, train, target, scoring="neg_mean_squared_error", cv=5)))
print("linear regression score: ", score)

cv = 5 #number of folds in cross-validation

alphas = np.logspace(-5,2,20)
scores = np.zeros((len(alphas),cv))
scores_mu = np.zeros(len(alphas))
scores_sigma = np.zeros(len(alphas))

for i in range(0,len(alphas)):
    model = Ridge(alpha=alphas[i])
    scores[i,:] = np.sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = cv))
    scores_mu[i] = np.mean(scores[i,:])
    scores_sigma[i] = np.std(scores[i,:])

# figure(figsize(8,4))
#for i in range(0,cv):
#    plot(alphas,scores[:,i], 'b--', alpha=0.5)
# plot(alphas,scores_mu,'c-',lw=3, alpha=0.5, label = "Ridge")
# fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),
#              np.array(scores_mu)+np.array(scores_sigma),color="c",alpha=0.5)

print("best score in Ridge: ",min(scores_mu))

for i in range(0,len(alphas)):
    model = Lasso(alpha=alphas[i])
    scores[i,:] = np.sqrt(-cross_val_score(model, train, target,scoring="neg_mean_squared_error", cv = cv))
    scores_mu[i] = np.mean(scores[i,:])
    scores_sigma[i] = np.std(scores[i,:])

# plot(alphas,scores_mu,'g-',lw=3, alpha=0.5, label="Lasso")
# fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),
#              np.array(scores_mu)+np.array(scores_sigma),color="g",alpha=0.5)

plt.xscale("log")
plt.xlabel("alpha", size=20)
plt.ylabel("rmse", size=20)
plt.legend(loc=2)

print("best score in Lasso: ",min(scores_mu))

model = Lasso(alpha=1e-4)
model.fit(train,target)
preds = model.predict(test)
preds = 10**preds

solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
print(train.describe())
solution.to_csv("../solutions/submit.csv", index = False)
