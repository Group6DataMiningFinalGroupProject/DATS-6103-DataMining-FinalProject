#Model Functions used for training and predicting the data
#import Necessary Packages
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from itertools import combinations
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
seed = 100
num_trees = 500
#Multi Class Logistic Regression
def Logisticregression(train_feature_data,train_label_data,test_feature_data):
    '''
    mlg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',max_iter = 1000)
    lg_clf = mlg.fit(train_feature_data, train_label_data)
    lr_mod_file = 'Logistic_regression.sav'
    joblib.dump(lg_clf, lr_mod_file)
    '''
    lg_clf = joblib.load('Logistic_regression.sav')
    pred_lg = lg_clf.predict(test_feature_data)
    return pred_lg

#SVM Linear Kernel
def SVM_linear(train_feature_data,train_label_data,test_feature_data):
    '''
    svm_clf = SVC(kernel='linear', max_iter=1000)
    svm_clf.fit(train_feature_data, train_label_data)
    svml_mod_file = 'SVM_Linear.sav'
    joblib.dump(svm_clf, svml_mod_file)
    '''
    svm_clf = joblib.load('SVM_Linear.sav')
    pred_svml = svm_clf.predict(test_feature_data)
    return pred_svml

#SVM Non Linear Kernel
#Grid Search - Hyper parameter tuning - Parameter - "C" and "gamma" values
params_dict = {"C": np.logspace(-1, 3, 10), "gamma": np.linspace(0.0001, 10, 10)}
svm = SVC(kernel="rbf", max_iter = 1000)
def SVM_Non_linear(train_feature_data,train_label_data,test_feature_data):
    '''
    search = GridSearchCV(estimator=svm, param_grid=params_dict)
    search.fit(train_feature_data, train_label_data)
    print("Best parameter values:", search.best_params_)
    Svm_mod_file = 'Svm_mod_file_1000.sav'
    joblib.dump(search, Svm_mod_file)
    loaded_model = joblib.load('Svm_mod_file_1000.sav')
    best_svm = loaded_model.best_estimator_
    best_svm.fit(train_feature_data, train_label_data)
    svmnl_mod_file = 'SVM_nonlinear.sav'
    joblib.dump(best_svm, svmnl_mod_file)
    '''
    best_svm = joblib.load('SVM_nonlinear.sav')
    pred_svmnl = best_svm.predict(test_feature_data)
    return pred_svmnl

#Random Forest Model
def RandomForest(train_feature_data,train_label_data,test_feature_data):
    '''
    model_rf = RandomForestClassifier(n_estimators=500)
    model_rf.fit(train_feature_data, train_label_data)
    rf_mod_file = 'Random_Forest.sav'
    joblib.dump(model_rf, rf_mod_file)
    '''
    model_rf = joblib.load('Random_Forest.sav')
    pred_rf = model_rf.predict(test_feature_data)
    return pred_rf

#Decision Tree Model
def DecisionTree(train_feature_data,train_label_data,test_feature_data):
    '''
    model_DT = DecisionTreeClassifier(random_state=seed)
    model_DT.fit(train_feature_data, train_label_data)
    DT_mod_file = 'Decision_Tree.sav'
    joblib.dump(model_DT, DT_mod_file)
    '''
    model_DT = joblib.load('Decision_Tree.sav')
    pred_DT = model_DT.predict(test_feature_data)
    return pred_DT

#KNN Model
def Knnclassifier(train_feature_data,train_label_data,test_feature_data):
    '''
    model_KNN = KNeighborsClassifier()
    model_KNN.fit(train_feature_data, train_label_data)
    KNN_mod_file = 'Knn_model.sav'
    joblib.dump(model_KNN, KNN_mod_file)
    '''
    model_KNN = joblib.load('Knn_model.sav')
    pred_KNN = model_KNN.predict(test_feature_data)
    return pred_KNN

#NaiveBayes Model
def NaiveBayes(train_feature_data,train_label_data,test_feature_data):
    '''
    model_NB = GaussianNB()
    model_NB.fit(train_feature_data, train_label_data)
    NB_mod_file = 'NB_model.sav'
    joblib.dump(model_NB, NB_mod_file)
    '''
    model_NB = joblib.load('NB_model.sav')
    pred_NB = model_NB.predict(test_feature_data)
    return pred_NB

#Ensemble Model - Hard Voting
#Best combination of Model for Hard Voting
#To save computational time,preloaded statistics are used
'''
LR = LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial',max_iter = 1000)
KNN = KNeighborsClassifier()
DT = DecisionTreeClassifier(random_state=seed)
RF = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
NB =  GaussianNB()
SVM_Lin = SVC(random_state=seed, kernel='rbf', max_iter = 1000, C = 0.1, gamma = 0.0001)
SVM_nonlinear = SVC(kernel='linear', max_iter = 1000)
Algm = [('lr', LR),('RF', RF),('KNN',KNN),('DT',DT),('NB',NB),('SVM_Lin',SVM_Lin),('SVM_nonlinear',SVM_nonlinear)]
combination= []
accuracy = 0
for i in range(len(Algm)+1):
    comb_1 = list(map(list,combinations(Algm,i)))
    combination.append(comb_1)
'''
def Best_voting(train_feature_data,train_label_data,test_feature_data,test_label_data):
    LR = LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial', max_iter=1000)
    KNN = KNeighborsClassifier()
    DT = DecisionTreeClassifier(random_state=seed)
    RF = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
    NB = GaussianNB()
    SVM_nonlinear = SVC(random_state=seed, kernel='rbf', max_iter=1000, C=0.1, gamma=0.0001)
    SVM_Lin = SVC(kernel='linear', max_iter=1000)
    '''
    for combo in combination[1:9]:
        for clas in combo:
            best_model = VotingClassifier(estimators=clas, voting='hard')
            best_model.fit(train_feature_data, train_label_data)
            preds = best_model.predict(test_feature_data)
            accuracy_model = accuracy_score(test_label_data, preds)
            if accuracy_model > accuracy:
                accuracy = accuracy_model
                best = clas
    print("Best Combination of Model for Hard Voting is ")
    print(best)
    print(accuracy)
    '''
    best = [('RF', RF),('SVM_Non',SVM_nonlinear)]
    return best

def ensemble_model(train_feature_data,train_label_data,test_feature_data,best):
    '''
    ensemble_mod = VotingClassifier(estimators=best, voting='hard')
    ensemble_mod.fit(train_feature_data, train_label_data)
    en_mod_file = 'ensemble_model.sav'
    joblib.dump(ensemble_mod, en_mod_file)
    '''
    ensemble_mod = joblib.load('ensemble_model.sav')
    pred_en = ensemble_mod.predict(test_feature_data)
    return pred_en













