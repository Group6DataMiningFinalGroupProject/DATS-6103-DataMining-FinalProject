import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit
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
from PyQt5.QtGui import QIcon
from Image_Download import download_prep
from Feature_Extraction import hog
from PyQt5 import QtWidgets, QtCore
import sys


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Live Data Prediction'
        self.left = 50
        self.top = 50
        self.width = 1066
        self.height = 740
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(20, 160, 1011, 271))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label.setText(" ")
        self.label_1 = QtWidgets.QLabel(self)
        self.label_1.setGeometry(QtCore.QRect(500, 130, 100, 20))
        self.label_1.setAlignment(QtCore.Qt.AlignCenter)
        self.label_1.setObjectName("label")
        self.label_1.setText("RESULTS")
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.getText()
        self.show()

    def getText(self):
        warnings.filterwarnings("ignore")
        # reading CSV as Numpy array - Pre extracted values
        train_feature_data = np.genfromtxt("./train_feature_data.csv", delimiter=',')
        test_feature_data = np.genfromtxt("./test_feature_data.csv", delimiter=',')
        train_label_data = np.genfromtxt("./train_label_data.csv", delimiter=',')
        test_label_data = np.genfromtxt("./test_label_data.csv", delimiter=',')
        seed = 100
        num_trees = 100
        lg_clf = joblib.load('./Logistic_regression.sav')
        svm_clf = joblib.load('./SVM_Linear.sav')
        best_svm = joblib.load('./SVM_nonlinear.sav')
        model_rf = joblib.load('./Random_Forest.sav')
        model_DT = joblib.load('./Decision_Tree.sav')
        model_KNN = joblib.load('./Knn_model.sav')
        model_NB = joblib.load('./NB_model.sav')
        ensemble_mod = joblib.load('./ensemble_model.sav')
        Loaded_Algm = [lg_clf, svm_clf, best_svm, model_rf, model_DT, model_KNN, model_NB, ensemble_mod]
        text, okPressed = QInputDialog.getText(self, "Live Data Prediction", "Enter the URL:", QLineEdit.Normal, "")
        if okPressed and text != '':
            Test_val = ["Test_01", text]
            im_val = pd.Series(Test_val)
            loc = "Test_image"
            download_prep(im_val, loc)
            if os.path.exists('./Resized_image/Test_image/' + str(im_val.iloc[0]) + '.jpg'):
                his = hog(io.imread('./Resized_image/Test_image/' + str(im_val.iloc[0]) + '.jpg'))
                test_data = his
                test_data = np.array([test_data])
                predics = []
                Algm_used = pd.Series(["LR", "SVM_LINEAR", "SVM_NON_LINEAR", "RF", "DT", "KNN", "NB", "ENSEMBLE"])
                for i in Loaded_Algm:
                    preds = i.predict(test_data)
                    predics.append(preds)
                predic = pd.Series(predics)
                Pred_out = pd.concat([Algm_used, predic], axis=1)
                self.label.setText(str(Pred_out))
                os.remove('./Resized_image/Test_image/' + str(im_val.iloc[0]) + '.jpg')
            else:
                self.label.setText("Please check Image URL - Unable to load either link is broken or it doesnt have image file")



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    liv = App()
    liv.show()
    sys.exit(app.exec_())
