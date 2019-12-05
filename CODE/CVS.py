# Load the Dataset file
# Import Packages
# train.csv - datafile contains details image details - id,URL and landmarkid
# Top 10 sampled landmark details are extracted for analysis
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
# import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import warnings
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CVS(object):
    def setupUi(self, CVS):
        CVS.setObjectName("CVS")
        CVS.resize(1072, 937)
        self.groupBox = QtWidgets.QGroupBox(CVS)
        self.groupBox.setGeometry(QtCore.QRect(-1, 9, 1071, 921))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 90, 1031, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 140, 1031, 41))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 180, 1031, 311))
        self.label_3.setAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(20, 490, 1031, 381))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(20, 870, 1031, 41))
        self.label_5.setObjectName("label_5")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(480, 30, 112, 34))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.cvs)

        self.retranslateUi(CVS)
        QtCore.QMetaObject.connectSlotsByName(CVS)

    def retranslateUi(self, CVS):
        _translate = QtCore.QCoreApplication.translate
        CVS.setWindowTitle(_translate("CVS", "Landmark Recognition - Cross-Validation Score"))
        self.groupBox.setTitle(_translate("CVS", "Click to get Results"))
        self.pushButton.setText(_translate("CVS", "Result"))
        self.label.setText(_translate("CVS", ""))
        self.label_2.setText(_translate("CVS", ""))
        self.label_3.setText(_translate("CVS", ""))
        self.label_4.setText(_translate("CVS", ""))
        self.label_5.setText(_translate("CVS", ""))

    def cvs(self):
        warnings.filterwarnings("ignore")
        # reading CSV as Numpy array - Pre extracted values
        train_feature_data = np.genfromtxt("./train_feature_data.csv", delimiter=',')
        test_feature_data = np.genfromtxt("./test_feature_data.csv", delimiter=',')
        train_label_data = np.genfromtxt("./train_label_data.csv", delimiter=',')
        test_label_data = np.genfromtxt("./test_label_data.csv", delimiter=',')
        # Modelling
        # Cross Validation Score for each Model
        # Below code is used for cross validation Purpose
        # Cross validated score is loaded directly to save computational time
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
        Result_df = pd.read_csv("./Cross_Validation_Data.csv")
        self.label_2.setText("Model CV Score")
        self.label_3.setText(str(Result_df))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_CVS()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

