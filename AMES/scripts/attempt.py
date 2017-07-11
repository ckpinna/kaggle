import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv("../data/test.csv")

target = df_train['SalePrice']
df_train = df_train.drop('SalePrice', 1)
CompleteSet = pd.concat((df_train, df_test))
print(df_train)
print(target.describe())

print(np.shape(df_test))
print(np.shape(target))
print(np.shape(CompleteSet))

CompleteSet = pd.get_dummies(CompleteSet)
CompleteSet = CompleteSet.fillna(CompleteSet.mean())

newTrain = CompleteSet[:df_train.shape[0]]
newTest = CompleteSet[df_train.shape[0]:]
target = np.log(target)

model = LinearRegression()
score = np.mean(np.sqrt(-cross_val_score(model, newTrain, target, scoring="neg_mean_squared_error", cv=5)))
print("linear regression score: ", score)

cv = 5

alphas = np.logspace(-5,2,20)
scores = np.zeros((len(alphas),cv))
scores_mu = np.zeros(len(alphas))
scores_sigma = np.zeros(len(alphas))

for i in range(0,len(alphas)):
    model = Ridge(alpha=alphas[i])
    scores[i,:] = np.sqrt(-cross_val_score(model, newTrain, target,scoring="neg_mean_squared_error", cv = cv))
    scores_mu[i] = np.mean(scores[i,:])
    scores_sigma[i] = np.std(scores[i,:])

# figure(figsize(8,4))
#for i in range(0,cv):
#    plot(alphas,scores[:,i], 'b--', alpha=0.5)
plt.plot(alphas,scores_mu,'c-',lw=3, alpha=0.5, label = "Ridge")
plt.fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),
             np.array(scores_mu)+np.array(scores_sigma),color="c",alpha=0.5)
print("best score in Ridge: ",min(scores_mu))

for i in range(0,len(alphas)):
    model = Lasso(alpha=alphas[i])
    scores[i,:] = np.sqrt(-cross_val_score(model, newTrain, target,scoring="neg_mean_squared_error", cv = cv))
    scores_mu[i] = np.mean(scores[i,:])
    scores_sigma[i] = np.std(scores[i,:])

plt.plot(alphas,scores_mu,'g-',lw=3, alpha=0.5, label="Lasso")
plt.fill_between(alphas,np.array(scores_mu)-np.array(scores_sigma),
             np.array(scores_mu)+np.array(scores_sigma),color="g",alpha=0.5)

plt.xscale("log")
plt.xlabel("alpha", size=20)
plt.ylabel("rmse", size=20)
plt.legend(loc=2)

print("best score in Lasso: ",min(scores_mu))

# plt.show()

model = Lasso(alpha=1e-4)
model.fit(newTrain,target)
preds = model.predict(newTest)
preds = 10**preds
solution = pd.DataFrame({"id":newTest.Id, "SalePrice":preds})
print(solution)
solution.to_csv("../solutions/submit.csv", index = False)
