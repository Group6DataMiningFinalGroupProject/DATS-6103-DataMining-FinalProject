#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import seaborn as sns
train  = pd.read_csv("./Data/train.csv")
val = train["landmark_id"].value_counts()
val = pd.DataFrame(val)
val["Landmark_id"] = val.index
val = val.reset_index(drop = True)
val = val.rename(columns = {"Landmark_id" : "Landmark_id","landmark_id" : "Frequency"})
top_10_landmark_id = list(val.iloc[0:10,]["Landmark_id"])
top_df = pd.DataFrame()
top_df = train[train["landmark_id"].isin(top_10_landmark_id)]
top_df = top_df.reset_index(drop = True)
xTrain, xTest = train_test_split(top_df, test_size = 0.3, random_state = 0)
import numpy as np
import cv2
bin_n = 16
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
train_feature = []
train_labels = []
test_feature = []
test_labels = []
errored_train = []
errored_test = []
from Feature_Extraction import hog
#Train Images
for i in range(len(xTrain)):
    im_info_train = xTrain.iloc[i]
    try:
        if os.path.exists('./Resized_image/Train_image/'+str(im_info_train.iloc[0])+'.jpg'):
            his = hog(io.imread('./Resized_image/Train_image/'+str(im_info_train.iloc[0])+'.jpg'))
            train_feature.append(his)
            train_labels.append(im_info_train.iloc[2])
    except:
        print("Train ", im_info_train.iloc[0])
        errored_train.append(im_info_train.iloc[0])
#Test Images
for i in range(len(xTest)):
    im_info_test = xTest.iloc[i]
    try:
        if os.path.exists('./Resized_image/Test_image/'+str(im_info_test.iloc[0])+'.jpg'):
            his = hog(io.imread('./Resized_image/Test_image/'+str(im_info_test.iloc[0])+'.jpg'))
            test_feature.append(his)
            test_labels.append(im_info_test.iloc[2])
    except:
        print("Test ",im_info_test.iloc[0])
        errored_test.append(im_info_test.iloc[0])
print("Below are the errored image ID which is not present in given folder")
print("Missing Files from Train set : ",errored_train )
print("Missing Files from Test set : ",errored_test )
print("Feature Details are Extracted for all images in Train and Test dataset")

