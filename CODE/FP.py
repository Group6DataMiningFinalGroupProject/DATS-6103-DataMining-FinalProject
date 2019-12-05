from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Freqplot(object):
    def setupUi(self, Freqplot):
        Freqplot.setObjectName("Freqplot")
        Freqplot.resize(1038, 615)
        self.label = QtWidgets.QLabel(Freqplot)
        self.label.setGeometry(QtCore.QRect(17, 8, 1001, 591))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("./myplot.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")

        self.retranslateUi(Freqplot)
        QtCore.QMetaObject.connectSlotsByName(Freqplot)

    def retranslateUi(self, Freqplot):
        _translate = QtCore.QCoreApplication.translate
        Freqplot.setWindowTitle(_translate("Freqplot", "Landmark Recognition - Frequency Plot"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Freqplot()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())