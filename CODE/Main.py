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
#%%Download images and save in Train and Test Folders
#Data downloaded Already, so below commands are hashed out.
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
#`````````````````````````````````````````````````````````````````
#%%Feature Extraction
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
print("Feature Details are Extracted for all images in Train and Test dataset")
print(" Below are the errored image ID which is not present in given folder")
print("Missing Files from Train set : ",errored_train )
print("Missing Files from Test set : ",errored_test )
#%% Numpy Array for easy computation
#Train image
train_feature_data = np.float32(train_feature)
train_label_data = np.float32(train_labels)
#Test image
test_feature_data = np.float32(test_feature)
test_label_data = np.float32(test_labels)