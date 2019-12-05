#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#%%PRE-PROCESSING - DOWNLOADING IMAGES
# Load the Dataset file
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
train  = pd.read_csv("./Data/train.csv")
val = train["landmark_id"].value_counts()
#Frq = train.groupby("landmark_id").count().sort_values("id", ascending=False)
#Frq["id"].iloc[:10]
print("Original Train dataset is loaded")
print(" The Total number of observations are ", train.shape[0])
print(" Datafile contains ", train.columns)
print("Total Number of landmark classes available in original train file :",len(train["landmark_id"].unique()))
val = pd.DataFrame(val)
val["Landmark_id"] = val.index
val = val.reset_index(drop = True)
val = val.rename(columns = {"Landmark_id" : "Landmark_id","landmark_id" : "Frequency"})
print("Top 10 sampled data [Frequency along with Landmark_id]",val.iloc[0:10,] )
top_10_landmark_id = list(val.iloc[0:10,]["Landmark_id"])
top_df = pd.DataFrame()
top_df = train[train["landmark_id"].isin(top_10_landmark_id)]
top_df = top_df.reset_index(drop = True)
print(" Total Number of Observations in sampled data : ", top_df.shape[0] )
#%%Frequency Plot on Sampled dataset
top_df["landmark_id"].value_counts().head(10).plot('bar')
plt.xlabel('Landmark_id')
plt.ylabel('Frequency')
plt.title('Frequency Plot - top 10 Sampled Data')
plt.show()
#%%Splitting the Dataset into Train and test set
#Dataset is split 70% and 30% Ratio
xTrain, xTest = train_test_split(top_df, test_size = 0.3, random_state = 0)
print("Number of observations in each split is given as ")
print(" XTrain :" , xTrain.shape[0])
print(" XTest  :" , xTest.shape[0])
from skimage import io
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
new_size = (256, 256)
errored_id = []
def download_prep(im_info,loc):
    try:
        response=requests.get(im_info.iloc[1], stream=True)
        open('./Resized_image1/'+str(im_info.iloc[0])+'.jpg','wb').write(response.content)
        img1 = io.imread('./Resized_image1/'+str(im_info.iloc[0])+'.jpg')
        io.imsave('./Resized_image/'+str(loc)+'/'+str(im_info.iloc[0])+'.jpg',img_as_ubyte(np.array(resize(img1,new_size,mode='reflect', anti_aliasing = True,anti_aliasing_sigma=None))))
        os.remove('./Resized_image1/'+str(im_info.iloc[0])+'.jpg')
    except:
        print(im_info.iloc[0])
        errored_id.append(im_info.iloc[0])
#%%Download images and save in Train and Test Folders
#To save computational time,Data downloaded Already, so below commands are hashed out.
#````````````````````````````````````````````````````````
#Train Dataset:
from Image_Download import download_prep
start_time = time.time()
for i in range(len(xTrain)):
    if (i % 100 == 0):
        print("Time Taken for loading ", i ,"images is " , (time.time() - start_time), "Seconds")
    im_info = xTrain.iloc[i]
    loc = "Train_image"
    download_prep(im_info,loc)
#Test Dataset
for i in range(len(xTest)):
    if (i % 100 == 0):
        print( "Time Taken for loading ", i ,"images is " , (time.time() - start_time), "Seconds")
    im_info = xTest.iloc[i]
    loc = "Test_image"
    download_prep(im_info,loc)
#``````````````````````````````````````````````````


