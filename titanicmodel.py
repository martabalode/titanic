#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 11:05:07 2019

@author: Marta
"""

import pandas as pd
titanic = pd.read_csv('train.csv')
titanic.head()

#####EXPLORE THE DATA#####
#draw a histogram
from matplotlib import pyplot as plt
titanic["Age"].hist()
titanic["Pclass"].hist()
titanic["Survived"].hist()

#0=male, 1=female
titanic["Sex"] = pd.factorize(titanic["Sex"])[0]
gendergrouped = titanic.groupby("Sex")["Survived"].sum()
gendergrouped.plot.bar()
#groupbyclass
classgrouped = titanic.groupby("Pclass")["Survived"].sum()
classgrouped.plot.bar()
#groupbyage
agegrouped = titanic.groupby("Age")["Survived"].sum()
agegrouped.plot.bar()
plt.axis([0, 85, 0, 50])

#fill missing values
titanic['Age'].fillna(titanic["Age"].mean(), inplace = True)
#binning
titanic['Age'].cut(10)   # cuts by value range
titanic.plot.scatter("Age", "Survived")


#####BUILD A LOGISTIC REGRESSION MODEL#####
from sklearn.linear_model import LogisticRegression

y = titanic["Survived"] #define the y value
X = titanic[["Age", "Sex", "Pclass"]] #define x values

m = LogisticRegression(C=1e5) #name a model
m.fit(X, y) #apply the model to the predifined Xs and Ys
print(m.score(X, y)) #print the accuracy

dir(m)
m.coef_ #show the weights of each of Xs
m.intercept_ #show the intercept

leo = [[24, 0, 3]] #make the same shape as x (matrix) with double []
m.predict_proba(leo) #calculates both probability for notsurviving (0) and for surviving (1)
#create a new dataframe with predicted survival probabilities for all data points
forall = m.predict_proba(X)
forall = pd.DataFrame(forall)
#Rename the 0 and 1 columns to probability of survival
forall.rename(columns={0:'Probability for not surviving',
                       1: "Probability for surviving"}, inplace=True)
titanic = titanic.join(forall) #Merges 2 dataframes on the index



#####MODEL EVALUATION#####
###Calculate precision and recall###

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

#create a function that would compare all values from column "probability for surviving" to 0.5
#set a threshold under which all values get assigned a 0, and above which all get 1 = 0.5
def surviving(p):
    if p >= 0.50:
        return 1
    else:
        return 0
#make a new column of predicted values (probability for surviving) rounded to 1 or 0
titanic["ypred"] = titanic["Probability for surviving"].apply(surviving)

#Y predicted are the above; Y True are the left labels. 
#[0,0; 0,1]
#[1,0; 1,1]
#0,0= True Negative
#1,1= True Positive
confusion_matrix(y_pred = titanic["ypred"], y_true = titanic["Survived"])
precision_score(y_pred = titanic["ypred"], y_true = titanic["Survived"])
recall_score(y_pred = titanic["ypred"], y_true = titanic["Survived"])
accuracy_score(y_pred = titanic["ypred"], y_true = titanic["Survived"])

###ROC Curve###
y_true = titanic["Survived"]
y_prob = m.predict_proba(X)

import scikitplot as skplt
import matplotlib.pyplot as plt
skplt.metrics.plot_roc(y_true, y_prob, title = "ROC Curve",
                       plot_micro = False, plot_macro = False,
                       classes_to_plot = 1)



#####MODEL VALIDATION#####
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=0.2,
                                                random_state=42,
                                                stratify= y)
#apply the model to the Train data
from sklearn.linear_model import LogisticRegression
m = LogisticRegression()
m.fit(Xtrain, ytrain)
m.score(Xtrain, ytrain)
#to check how it performs on test data
m.score(Xtest, ytest)

#cross validation
from sklearn.model_selection import cross_val_score
crossvalscore = cross_val_score(X=Xtrain, y=ytrain, estimator= m, cv=5)

#bootstrapping
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
#for each iteration from the 1000 it draws a sample the same size as 
#Xtrain and ytrain, but as the datapoints are put back after each drawing
#the Xb and yb samples differ (because can contain the same datapoint many times)
m = LogisticRegression()
boots = []
for i in range(1000):
    Xb, yb = resample(Xtrain, ytrain)
    m.fit(Xb, yb)
    score = m.score(Xb, yb)
    boots.append(score)
    print(i, score)

#Confidence intervals
boots.sort() #first sort
ci80 = boots[100:-100] #cuts off 10% from both sides= 20%. 80% CI
print(f"80% confidence interval: {ci80[0]:5.2} -{ci80[-1]:5.2}") #5.2 means= 5 digits and 2 digits after the comma
#CI for 90%
ci90 = boots[50:-50] #cuts off 5% from both sides
print(f"90% confidence interval: {ci90[0]:5.2} -{ci90[-1]:5.2}")
#CI for 95%
ci95 = boots[25:-25] #cuts off 2.5% from both sides
print(f"95% confidence interval: {ci95[0]:5.2} -{ci95[-1]:5.2}")
#CI for 99%.
ci99 = boots[5:-5] #cuts off 0.5% from both sides
print(f"99% confidence interval: {ci99[0]:5.2} -{ci99[-1]:5.2}")








#### TUNE HYPERPARAMETERS ####


##grid search for logistic regression
from sklearn.model_selection import GridSearchCV
from pprint import pprint

X = Xtrain
y = ytrain
#put the specific model in the GridSearchCV (either random forest, logreg or svm)
gridLR = GridSearchCV(m, #model m=LR defined above
        param_grid={'C': [1.0, 0.1, 0.01, 0.001]}) #define hyperparameters which could give the best accuracies 

gridLR.fit(X, y) #fit the grid to the training data
print("all scores      :")
pprint(gridLR.cv_results_)
gridLR.best_score_ #best score gives the average accuracy score from the cross validation (how many models were created)
gridLR.best_params_ #find the best parameters that give the highest accuracy
m = LogisticRegression(C= 1.0) #put them into the model
m.fit(X, y) #fit the model with the adjusted parameters
m.score(X, y) #see how it performs
m.score(Xtest, ytest) #see how it performs on test data


##grid search for random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier() #define model
params = {
    'max_depth': [2, 3, 4, 5, 6],
    'n_estimators': [1, 3, 5, 7, 10, 15, 20]}
gRF = GridSearchCV(rf, param_grid=params)
gRF.fit(X, y)
gRF.score(X, y)
gRF.best_score_
gRF.best_params_
rf = RandomForestClassifier(max_depth = 6, n_estimators = 20)
rf.fit(X, y)
rf.score(X, y)
rf.score(Xtest, ytest)

##grid search for svm
#converting the data to binary might help the accuracy
#convert age to a binary variable
agebins = ["0-16", "16-50", "50+"]
Xtrain["Ageintervals"] = pd.cut(Xtrain["Age"], [0, 16, 50, 100], labels = agebins)
dummies = pd.get_dummies(Xtrain["Ageintervals"])
Xtrain1 = pd.merge(Xtrain, dummies, left_index = True, right_index = True)
#convert the Pclass to binary
dummies = pd.get_dummies(Xtrain["Pclass"])
Xtrain1 = pd.merge(Xtrain1, dummies, left_index = True, right_index = True)
#drop the original columns
Xtrain1.drop(["Age", "Ageintervals", "Pclass"], axis = 1, inplace = True)
#grid search for svm
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from pprint import pprint

svc = svm.SVC() #define model= support vector machine
gridSVM = GridSearchCV(svc, 
        param_grid={'C': [1.0, 0.1, 0.01, 0.001], 'kernel':['linear', 'rbf']},
        scoring='accuracy', 
        n_jobs=1,
        cv=None)
gridSVM.fit(Xtrain1, y)
bestscoreSVM= grid.best_score_
gridSVM.best_params_
bestrf = gridSVM.best_estimator_ #Access the best model directly
bestrf.fit(Xtrain1, y)
bestrf.score(Xtrain1, y)
bestrf.score(Xtest1, ytest)
#modify the test data set as well
Xtest["Ageintervals"] = pd.cut(Xtest["Age"], [0, 16, 50, 100], labels = agebins)
dummies = pd.get_dummies(Xtest["Ageintervals"])
Xtest1 = pd.merge(Xtest, dummies, left_index = True, right_index = True)
#convert the Pclass to binary
dummies = pd.get_dummies(Xtest["Pclass"])
Xtest1 = pd.merge(Xtest1, dummies, left_index = True, right_index = True)
#drop the original columns
Xtest1.drop(["Age", "Ageintervals", "Pclass"], axis = 1, inplace = True)
Xtest.drop("Ageintervals", axis = 1, inplace = True)



#### MODELING PIPELINE ####
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm

model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('svc', svm.SVC(kernel='rbf', C=1.0)),])

model.fit(Xtest, ytest)
print(model.score(Xtest, ytest))




#### APPLY A RANDOM FOREST ####
#create a decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import os

m = DecisionTreeClassifier(max_depth=3)
m.fit(Xtrain, ytrain)
m.score(Xtrain, ytrain)

# create string in .dot format
tree = export_graphviz(m, out_file=None, 
                class_names=["0", "1"],
                feature_names=['Age', 'Sex', "Pclass"],
                impurity=False,
                filled=True)
open('titanic.dot', 'w').write(tree)

graph = graphviz.Source(tree)
# PNG conversion (tested on Ubuntu)
cmd = "dot -Tpng titanic.dot -o tree_graphviz.png"
os.system(cmd)


#### UPLOAD TO KAGGLE ####
predict = pd.read_csv('predict.csv')
predict["Sex"] = pd.factorize(predict["Sex"])[0]
predict['Age'].fillna(predict["Age"].mean(), inplace = True)
Xpredict = predict[["Age", "Sex", "Pclass"]]
ypredict = rf.predict(Xpredict)
predict.set_index("PassengerId", inplace = True)
predict["Survived"] = ypredict
submission = predict[["Survived"]]
submission.to_csv("submissionrf.csv")












