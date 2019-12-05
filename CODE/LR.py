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


class Ui_LR(object):
    def setupUi(self, LR):
        LR.setObjectName("LR")
        LR.resize(1072, 937)
        self.groupBox = QtWidgets.QGroupBox(LR)
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
        self.pushButton.clicked.connect(self.log_reg)

        self.retranslateUi(LR)
        QtCore.QMetaObject.connectSlotsByName(LR)

    def retranslateUi(self, LR):
        _translate = QtCore.QCoreApplication.translate
        LR.setWindowTitle(_translate("LR", "Landmark Recognition - Logistic Regression"))
        self.groupBox.setTitle(_translate("LR", "Click to get Results"))
        self.pushButton.setText(_translate("LR", "Result"))
        self.label.setText(_translate("LR", ""))
        self.label_2.setText(_translate("LR", ""))
        self.label_3.setText(_translate("LR", ""))
        self.label_4.setText(_translate("LR", ""))
        self.label_5.setText(_translate("LR", ""))

    def log_reg(self):
        warnings.filterwarnings("ignore")
        # reading CSV as Numpy array - Pre extracted values
        train_feature_data = np.genfromtxt("./train_feature_data.csv", delimiter=',')
        test_feature_data = np.genfromtxt("./test_feature_data.csv", delimiter=',')
        train_label_data = np.genfromtxt("./train_label_data.csv", delimiter=',')
        test_label_data = np.genfromtxt("./test_label_data.csv", delimiter=',')
        # Modelling
        seed = 100
        num_trees = 100
        # %% Modelling
        test_label_data_val = test_label_data.astype("str")
        class_names = np.unique(test_label_data_val)
        Model_comp = pd.DataFrame(columns=['Model', 'Accuracy_Score', 'Cohen Kappa Score'])
        from Model_Functions import Logisticregression
        pred_lg = Logisticregression(train_feature_data, train_label_data, test_feature_data)
        Accuracy_lg = accuracy_score(test_label_data, pred_lg) * 100
        conf_matrix_lg = confusion_matrix(test_label_data, pred_lg)
        df_cm_lg = pd.DataFrame(conf_matrix_lg, index=class_names, columns=class_names)
        cohen_lg = cohen_kappa_score(test_label_data, pred_lg)
        self.label.setText(" Model Evaluation Metrics - Logistic Regression ")
        self.label_2.setText("Accuracy Score : " + str(Accuracy_lg))
        self.label_3.setText("Confusion Matrix" + str(df_cm_lg))
        self.label_4.setText(" Classification Report \n" + str(classification_report(test_label_data, pred_lg)))
        self.label_5.setText(" Cohen Kappa Score " + str(cohen_lg))
        Model_comp = Model_comp.append({"Model": "Logistic Regression", "Accuracy_Score": Accuracy_lg, "Cohen Kappa Score": cohen_lg},
            ignore_index=True)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_LR()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

