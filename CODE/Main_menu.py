from PyQt5 import QtCore, QtGui, QtWidgets
from Model import Ui_Model
from FP import Ui_Freqplot
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time
from skimage import io
import os
import numpy as np


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 875)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 110, 970, 161))
        font = QtGui.QFont()
        font.setFamily("Cambria Math")
        font.setPointSize(9)
        self.groupBox.setFont(font)
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(20, 20, 241, 51))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.load_data)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 90, 241, 51))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.preproc)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_5.setGeometry(QtCore.QRect(340, 90, 241, 51))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.split)
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_6.setGeometry(QtCore.QRect(640, 90, 310, 51))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(self.click_model)
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_7.setGeometry(QtCore.QRect(340, 20, 241, 51))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(self.freqplt)
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_9.setGeometry(QtCore.QRect(640, 20, 255, 51))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(self.feat_ext)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 861, 61))
        font = QtGui.QFont()
        font.setFamily("Cambria Math")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_0 = QtWidgets.QLabel(self.centralwidget)
        self.label_0.setGeometry(QtCore.QRect(20, 290, 861, 31))
        self.label_0.setObjectName("label_0")
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_0.setFont(font)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 320, 861, 31))
        self.label_2.setObjectName("label_2")
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_2.setFont(font)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 350, 861, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 380, 861, 31))
        self.label_4.setObjectName("label_4")
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_4.setFont(font)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 480, 861, 331))
        self.label_7.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 410, 861, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 440, 861, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 919, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Landmark Recognition"))
        self.pushButton.setText(_translate("MainWindow", "LOAD DATA"))
        self.pushButton_2.setText(_translate("MainWindow", "PRE-PROCESSING"))
        self.pushButton_5.setText(_translate("MainWindow", "DATA SPLIT"))
        self.pushButton_6.setText(_translate("MainWindow", "MODELING and PREDICTION"))
        self.pushButton_7.setText(_translate("MainWindow", "FREQUENCY PLOT"))
        self.pushButton_9.setText(_translate("MainWindow", "FEATURE EXTRACTION"))
        self.label.setText(_translate("MainWindow", "LANDMARK RECOGNITION"))
        self.label_0.setText(_translate("MainWindow", ""))
        self.label_2.setText(_translate("MainWindow", ""))
        self.label_3.setText(_translate("MainWindow", ""))
        self.label_4.setText(_translate("MainWindow", ""))
        self.label_7.setText(_translate("MainWindow", ""))
        self.label_5.setText(_translate("MainWindow", ""))
        self.label_6.setText(_translate("MainWindow", ""))

    def click_model(self):
        Dialog1 = QtWidgets.QDialog()
        ui1 = Ui_Model()
        ui1.setupUi(Dialog1)
        Dialog1.show()
        Dialog1.exec_()

    def load_data(self):
        train = pd.read_csv("./Data/train.csv")
        val = train["landmark_id"].value_counts()
        # Frq = train.groupby("landmark_id").count().sort_values("id", ascending=False)
        # Frq["id"].iloc[:10]
        self.label_0.setText("Original Train dataset is loaded")
        self.label_2.setText("The Total number of observations are" + str(train.shape[0]))
        self.label_3.setText("Datafile contains" + str(train.columns))
        self.label_4.setText("Total Number of landmark classes available in original train file :" + str(len(train["landmark_id"].unique())))
        val = pd.DataFrame(val)
        val["Landmark_id"] = val.index
        val = val.reset_index(drop=True)
        val = val.rename(columns={"Landmark_id": "Landmark_id", "landmark_id": "Frequency"})
        self.label_6.setText("Top 10 sampled data [Frequency along with Landmark_id]")
        self.label_7.setText(str(val.iloc[0:10, ]))

        top_10_landmark_id = list(val.iloc[0:10, ]["Landmark_id"])
        top_df = train[train["landmark_id"].isin(top_10_landmark_id)]
        top_df = top_df.reset_index(drop=True)
        self.label_5.setText("Total Number of Observations in sampled data : " + str(top_df.shape[0]))

    def freqplt(self):
        Dialog = QtWidgets.QDialog()
        ui = Ui_Freqplot()
        ui.setupUi(Dialog)
        Dialog.show()
        Dialog.exec_()

    def split(self):
        train = pd.read_csv("./Data/train.csv")
        val = train["landmark_id"].value_counts()
        # Frq = train.groupby("landmark_id").count().sort_values("id", ascending=False)
        # Frq["id"].iloc[:10]
        val = pd.DataFrame(val)
        val["Landmark_id"] = val.index
        val = val.reset_index(drop=True)
        val = val.rename(columns={"Landmark_id": "Landmark_id", "landmark_id": "Frequency"})
        top_10_landmark_id = list(val.iloc[0:10, ]["Landmark_id"])
        top_df = train[train["landmark_id"].isin(top_10_landmark_id)]
        top_df = top_df.reset_index(drop=True)
        # splitting the Dataset into Train and test set
        # Dataset is split 70% and 30% Ratio
        xTrain, xTest = train_test_split(top_df, test_size=0.3, random_state=0)
        self.label_0.setText("Number of observations in each split is given as ")
        self.label_2.setText(" XTrain :" + str(xTrain.shape[0]))
        self.label_3.setText(" XTest  :" + str(xTest.shape[0]))
        self.label_4.setText(" ")
        self.label_5.setText(" ")
        self.label_6.setText(" ")
        self.label_7.setText(" ")

    def preproc(self):
        self.label_0.setText("The data has been loaded, image files have been downloaded")
        self.label_2.setText("and the images have been resized to a single aspect ratio")
        self.label_3.setText("to maintain uniformity.")
        self.label_4.setText(" ")
        self.label_5.setText(" ")
        self.label_6.setText(" ")
        self.label_7.setText(" ")

    def feat_ext(self):
        train = pd.read_csv("./Data/train.csv")
        val = train["landmark_id"].value_counts()
        # Frq = train.groupby("landmark_id").count().sort_values("id", ascending=False)
        # Frq["id"].iloc[:10]
        # print("Original Train dataset is loaded")
        # print(" The Total number of observations are ", train.shape[0])
        # print(" Datafile contains ", train.columns)
        # print("Total Number of landmark classes available in original train file :", len(train["landmark_id"].unique()))
        val = pd.DataFrame(val)
        val["Landmark_id"] = val.index
        val = val.reset_index(drop=True)
        val = val.rename(columns={"Landmark_id": "Landmark_id", "landmark_id": "Frequency"})
        # print("Top 10 sampled data [Frequency along with Landmark_id]", val.iloc[0:10, ])
        top_10_landmark_id = list(val.iloc[0:10, ]["Landmark_id"])
        # top_df = pd.DataFrame()
        top_df = train[train["landmark_id"].isin(top_10_landmark_id)]
        top_df = top_df.reset_index(drop=True)
        # print(" Total Number of Observations in sampled data : ", top_df.shape[0])
        # %%Frequency Plot on Sampled dataset
        # top_df["landmark_id"].value_counts().head(10).plot('bar')
        # plt.xlabel('Landmark_id')
        # plt.ylabel('Frequency')
        # plt.title('Frequency Plot - top 10 Sampled Data')
        # plt.show()
        # %%Splitting the Dataset into Train and test set
        # Dataset is split 70% and 30% Ratio
        xTrain, xTest = train_test_split(top_df, test_size=0.3, random_state=0)
        # print("Number of observations in each split is given as ")
        # print(" XTrain :", xTrain.shape[0])
        # print(" XTest  :", xTest.shape[0])
        # `````````````````````````````````````````````````````````````````
        # %%Feature Extraction
        # Feature takes aproximately 10 mins to complete for given dataset, so for demo purpose to load the data faster
        # Data is preloaded. below codes are used for extracting purpose
        '''
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
        '''
        # print("Feature Details are Extracted for all images in Train and Test dataset")

        # %% Numpy Array for easy computation
        # Train image
        '''
        train_feature_data = np.float32(train_feature)
        train_label_data = np.float32(train_labels)
        #Test image
        test_feature_data = np.float32(test_feature)
        test_label_data = np.float32(test_labels)
        #To save computational time while loading[Feature extraction for Xtrain and Xtest takes about 10 minutes], Data is loaded into csv and read as numpy
        np.savetxt("train_feature_data.csv",train_feature_data, delimiter=",")
        np.savetxt("test_feature_data.csv",test_feature_data, delimiter=",")
        np.savetxt("train_label_data.csv",train_label_data,delimiter = ',')
        np.savetxt("test_label_data.csv",test_label_data,delimiter = ',')
        '''
        # reading CSV as Numpy array - Pre extracted values
        train_feature_data = np.genfromtxt("./train_feature_data.csv", delimiter=',')
        test_feature_data = np.genfromtxt("./test_feature_data.csv", delimiter=',')
        train_label_data = np.genfromtxt("./train_label_data.csv", delimiter=',')
        test_label_data = np.genfromtxt("./test_label_data.csv", delimiter=',')
        self.label_0.setText(" ")
        self.label_2.setText("Features have been extracted successfully using HOG Classifier")
        self.label_3.setText(" ")
        self.label_4.setText(" ")
        self.label_5.setText(" ")
        self.label_6.setText(" ")
        self.label_7.setText(" ")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MV = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MV)
    MV.show()
    sys.exit(app.exec_())