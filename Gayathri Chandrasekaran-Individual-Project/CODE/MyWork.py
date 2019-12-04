#%%Load the Dataset file
#Import Packages
#train.csv - datafile contains details image details - id,URL and landmarkid
#Top 10 sampled landmark details are extracted for analysis
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
from skimage import io
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings("ignore")
#%%reading CSV as Numpy array - Pre extracted values
train_feature_data = np.genfromtxt("./train_feature_data.csv",delimiter = ',')
test_feature_data = np.genfromtxt("./test_feature_data.csv",delimiter = ',')
train_label_data = np.genfromtxt("./train_label_data.csv",delimiter = ',')
test_label_data = np.genfromtxt("./test_label_data.csv",delimiter = ',')
#%% Modelling
# Cross Validation Score for each Model
#Below code is used for cross validation Purpose
#Cross validated score is loaded directly to save computational time
seed = 100
num_trees = 100
'''
models = []
models.append(('LR', LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial',max_iter = 1000)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM Non Linear', SVC(random_state=seed, kernel='rbf', max_iter = 1000, C = 0.1, gamma = 0.0001)))
models.append(('SVM Linear',SVC(kernel='linear', max_iter = 1000)))
# 10-fold cross validation
scoring = "accuracy"
Result_df = pd.DataFrame(columns=['Classifier_Name','Cross_validation_Score'])
for classifi, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_val = cross_val_score(model, train_feature_data,train_label_data, cv=kfold, scoring=scoring)
    print(classifi,cv_val.mean())
    Result_df = Result_df.append({"Classifier_Name":classifi,"Cross_validation_Score":cv_val.mean()}, ignore_index= True)
Result_df.to_csv("Cross_Validation_Data.csv", index=False)
'''
Result_df = pd.read_csv("Cross_Validation_Data.csv")
print("Model CV Score")
print(Result_df)
#%% Modelling
test_label_data_val = test_label_data.astype("str")
class_names = np.unique(test_label_data_val)
Model_comp = pd.DataFrame(columns=['Model','Accuracy_Score','Cohen Kappa Score'])
#Logistic Regression
from Model_Functions import Logisticregression
pred_lg = Logisticregression(train_feature_data,train_label_data,test_feature_data)
Accuracy_lg = accuracy_score(test_label_data, pred_lg) * 100
conf_matrix_lg = confusion_matrix(test_label_data, pred_lg)
df_cm_lg = pd.DataFrame(conf_matrix_lg, index=class_names, columns=class_names)
cohen_lg = cohen_kappa_score(test_label_data, pred_lg)
print(" Model Evaluation Metrics - Logistic Regression ")
print("Accuracy Score : ",Accuracy_lg )
print("Confusion Matrix",df_cm_lg )
print(" Classification Report \n", classification_report(test_label_data, pred_lg) )
print(" Cohen Kappa Score ",cohen_lg )
Model_comp = Model_comp.append({"Model":"Logistic Regression","Accuracy_Score":Accuracy_lg,"Cohen Kappa Score":cohen_lg}, ignore_index= True)
#%%Random Forest
from Model_Functions import RandomForest
pred_rf = RandomForest(train_feature_data,train_label_data,test_feature_data)
Accuracy_rf = accuracy_score(test_label_data, pred_rf) * 100
conf_matrix_rf = confusion_matrix(test_label_data, pred_rf)
df_cm_rf = pd.DataFrame(conf_matrix_rf, index=class_names, columns=class_names)
cohen_rf = cohen_kappa_score(test_label_data, pred_rf)
print(" Model Evaluation Metrics - Random Forest ")
print("Accuracy Score : ",Accuracy_rf )
print("Confusion Matrix",df_cm_rf )
print(" Classification Report \n", classification_report(test_label_data, pred_rf) )
print(" Cohen Kappa Score ",cohen_rf )
Model_comp = Model_comp.append({"Model":"Random Forest","Accuracy_Score":Accuracy_rf,"Cohen Kappa Score":cohen_rf}, ignore_index= True)
#%%SVM Linear
from Model_Functions import SVM_linear
pred_svml = SVM_linear(train_feature_data,train_label_data,test_feature_data)
Accuracy_svml = accuracy_score(test_label_data, pred_svml) * 100
conf_matrix_svml = confusion_matrix(test_label_data, pred_svml)
df_cm_svml = pd.DataFrame(conf_matrix_svml, index=class_names, columns=class_names)
cohen_svml = cohen_kappa_score(test_label_data, pred_svml)
print(" Model Evaluation Metrics - SVM Linear Model")
print("Accuracy Score : ",Accuracy_svml )
print("Confusion Matrix",df_cm_svml )
print(" Classification Report \n", classification_report(test_label_data, pred_svml) )
print(" Cohen Kappa Score ",cohen_svml )
Model_comp = Model_comp.append({"Model":"SVM_Linear","Accuracy_Score":Accuracy_svml,"Cohen Kappa Score":cohen_svml}, ignore_index= True)
#%%SVM Non Linear
from Model_Functions import SVM_Non_linear
pred_svmnl = SVM_Non_linear(train_feature_data,train_label_data,test_feature_data)
Accuracy_svmnl = accuracy_score(test_label_data, pred_svmnl) * 100
conf_matrix_svmnl = confusion_matrix(test_label_data, pred_svmnl)
df_cm_svmnl = pd.DataFrame(conf_matrix_svmnl, index=class_names, columns=class_names)
cohen_svmnl = cohen_kappa_score(test_label_data, pred_svmnl)
print(" Model Evaluation Metrics - SVM Non Linear Model ")
print("Accuracy Score : ",Accuracy_svmnl )
print("Confusion Matrix",df_cm_svmnl )
print(" Classification Report \n", classification_report(test_label_data, pred_svmnl) )
print(" Cohen Kappa Score ",cohen_svmnl )
Model_comp = Model_comp.append({"Model":"SVM Non Linear","Accuracy_Score":Accuracy_svmnl,"Cohen Kappa Score":cohen_svmnl}, ignore_index= True)
#%%Decision tree
from Model_Functions import DecisionTree
pred_DT = DecisionTree(train_feature_data,train_label_data,test_feature_data)
Accuracy_DT = accuracy_score(test_label_data, pred_DT) * 100
conf_matrix_DT = confusion_matrix(test_label_data, pred_DT)
df_cm_DT = pd.DataFrame(conf_matrix_DT, index=class_names, columns=class_names)
cohen_DT = cohen_kappa_score(test_label_data, pred_DT)
print(" Model Evaluation Metrics - Decision Tree ")
print("Accuracy Score : ",Accuracy_DT )
print("Confusion Matrix",df_cm_DT )
print(" Classification Report \n", classification_report(test_label_data, pred_DT) )
print(" Cohen Kappa Score ",cohen_DT )
Model_comp = Model_comp.append({"Model":"Decision Tree","Accuracy_Score":Accuracy_DT,"Cohen Kappa Score":cohen_DT}, ignore_index= True)
#%%KNN Model
from Model_Functions import Knnclassifier
pred_KNN = Knnclassifier(train_feature_data,train_label_data,test_feature_data)
Accuracy_KNN = accuracy_score(test_label_data, pred_KNN) * 100
conf_matrix_KNN = confusion_matrix(test_label_data, pred_KNN)
df_cm_KNN = pd.DataFrame(conf_matrix_KNN, index=class_names, columns=class_names)
cohen_KNN = cohen_kappa_score(test_label_data, pred_KNN)
print(" Model Evaluation Metrics - KNN ")
print("Accuracy Score : ",Accuracy_KNN )
print("Confusion Matrix",df_cm_KNN )
print(" Classification Report \n", classification_report(test_label_data, pred_KNN) )
print(" Cohen Kappa Score ",cohen_KNN )
Model_comp = Model_comp.append({"Model":"KNN","Accuracy_Score":Accuracy_KNN,"Cohen Kappa Score":cohen_KNN}, ignore_index= True)
#%%NaiveBayes Model
from Model_Functions import NaiveBayes
pred_NB = NaiveBayes(train_feature_data,train_label_data,test_feature_data)
Accuracy_NB = accuracy_score(test_label_data, pred_NB) * 100
conf_matrix_NB = confusion_matrix(test_label_data, pred_NB)
df_cm_NB = pd.DataFrame(conf_matrix_NB, index=class_names, columns=class_names)
cohen_NB = cohen_kappa_score(test_label_data, pred_NB)
print(" Model Evaluation Metrics - Naive Bayes ")
print("Accuracy Score : ",Accuracy_NB )
print("Confusion Matrix",df_cm_NB )
print(" Classification Report \n", classification_report(test_label_data, pred_NB) )
print(" Cohen Kappa Score ",cohen_NB )
Model_comp = Model_comp.append({"Model":"Naive Bayes","Accuracy_Score":Accuracy_NB,"Cohen Kappa Score":cohen_NB}, ignore_index= True)
#%%Ensembling Algorithm
from Model_Functions import ensemble_model
from Model_Functions import Best_voting
Best_Alg = Best_voting(train_feature_data,train_label_data,test_feature_data,test_label_data)
pred_en = ensemble_model(train_feature_data,train_label_data,test_feature_data,Best_Alg)
Accuracy_en = accuracy_score(test_label_data, pred_en) * 100
conf_matrix_en = confusion_matrix(test_label_data, pred_en)
df_cm_en = pd.DataFrame(conf_matrix_en, index=class_names, columns=class_names)
cohen_en = cohen_kappa_score(test_label_data, pred_en)
print(" Model Evaluation Metrics - Ensemble Model ")
print("Accuracy Score : ",Accuracy_en )
print("Confusion Matrix",df_cm_en )
print(" Classification Report \n", classification_report(test_label_data, pred_en) )
print(" Cohen Kappa Score ",cohen_en )
Model_comp = Model_comp.append({"Model":"Ensemble(Hard voting)","Accuracy_Score":Accuracy_en,"Cohen Kappa Score":cohen_en}, ignore_index= True)
#%%Model Comparision
print(" Model Comparison ")
print(Model_comp)
fig = plt.figure()
ax  = plt.gca()
ax.bar(Model_comp["Model"],Model_comp["Accuracy_Score"])
fmtr = matplotlib.ticker.FixedFormatter(["LR","RF","SVM L","SVM NL","DT","KNN","NB","Ensemble"])
ax.xaxis.set_major_formatter(fmtr)
plt.xlabel('Model')
plt.ylabel('Accuracy Score')
plt.title('Model Evaluation - Accuracy Score')
plt.show()
#%%Prediction - Demo
from Image_Download import download_prep
from Feature_Extraction import hog
lg_clf = joblib.load('Logistic_regression.sav')
svm_clf = joblib.load('SVM_Linear.sav')
best_svm = joblib.load('SVM_nonlinear.sav')
model_rf = joblib.load('Random_Forest.sav')
model_DT = joblib.load('Decision_Tree.sav')
model_KNN = joblib.load('Knn_model.sav')
model_NB = joblib.load('NB_model.sav')
ensemble_mod = joblib.load('ensemble_model.sav')
Loaded_Algm = [lg_clf,svm_clf,best_svm,model_rf,model_DT,model_KNN,model_NB,ensemble_mod]
print("Enter the URL of the image ")
URL = str(input())
Test_val = ["Test_01",URL]
im_val = pd.Series(Test_val)
loc = "Test_image"
download_prep(im_val, loc)
if os.path.exists('./Resized_image/Test_image/'+str(im_val.iloc[0])+'.jpg'):
            his = hog(io.imread('./Resized_image/Test_image/'+str(im_val.iloc[0])+'.jpg'))
            test_data = his
            test_data = np.array([test_data])
            predics = []
            Algm_used = pd.Series(["LR", "SVM_LINEAR", "SVM_NON_LINEAR", "RF", "DT", "KNN", "NB", "ENSEMBLE"])
            for i in Loaded_Algm:
                preds = i.predict(test_data)
                predics.append(preds)
            predic = pd.Series(predics)
            Pred_out = pd.concat([Algm_used,predic],axis = 1)
            os.remove('./Resized_image/Test_image/'+str(im_val.iloc[0])+'.jpg')
else:
    print("Please check Image URL - Unable to load either link is broken or it doesnt have image file")











